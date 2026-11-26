"""Versioned Chroma retrieval. Embeddings are supplied explicitly, never by Chroma defaults."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

from src.agent.decide import RetrievalResult
from src.profiles.agent_text import display_tags

EMBEDDING = {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "max_seq_length": 256,
    "batch_size": 64,
    "seed": 20261001,
}


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_catalog(path):
    records = {}
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            mid = row["movie_id"]
            if type(mid) is not int or mid <= 0 or mid in records:
                raise ValueError("Catalog IDs must be unique positive integers")
            for key in ("title", "text"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing {key} for {mid}")
            for key in ("genres", "genome_tags", "user_tags"):
                if not isinstance(row.get(key), list) or not all(isinstance(x, str) for x in row[key]):
                    raise ValueError(f"Invalid {key} for {mid}")
            row["tags"] = display_tags(row["genome_tags"], row["user_tags"])
            year = re.search(r"\((\d{4})\)\s*$", row["title"])
            row["decade"] = f"{int(year[1]) // 10 * 10}s" if year else None
            records[mid] = row
    if not records:
        raise ValueError("Empty catalog")
    return records


class SentenceEncoder:
    def __init__(self, settings=None, device="cpu"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.settings = dict(settings or EMBEDDING)
        if not re.fullmatch(r"[0-9a-f]{40}", self.settings["revision"]):
            raise ValueError("Embedding revision must be an immutable Git ID")
        torch.manual_seed(self.settings["seed"])
        torch.set_num_threads(4)
        self.model = SentenceTransformer(self.settings["model_id"], revision=self.settings["revision"],
                                         device=device, trust_remote_code=False)
        self.model.max_seq_length = self.settings["max_seq_length"]

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.settings["batch_size"],
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)


def chroma_client(path="data/indexes/chroma", host=None, port=8000):
    import chromadb
    from chromadb.config import Settings
    settings = Settings(anonymized_telemetry=False)
    if host:
        from src.retrieval.chroma_http import bounded_http_client
        return bounded_http_client(host, port, settings)
    return chromadb.PersistentClient(path=str(path), settings=settings)


def close_chroma_client(client):
    """Stop a local Chroma system and release persistent index file handles.

    Chroma 1.3.5 does not expose a public ``Client.close()`` method. Its local
    client owns a shared ``System`` whose stop hook drops the Rust bindings;
    clearing the matching cache entry prevents a stopped system from being
    reused. This matters on Windows, where open HNSW files prevent temporary
    index directories from being removed.
    """
    identifier = getattr(client, "_identifier", None)
    systems = getattr(type(client), "_identifier_to_system", None)
    if identifier is None or not isinstance(systems, dict):
        return
    system = systems.get(identifier)
    if system is None:
        return  # Already closed, or not a shared-system client.
    system.stop()
    if systems.get(identifier) is system:
        del systems[identifier]


def ingest(client, documents, catalog_csv, manifest_path, encoder):
    """Create a new collection; publish its immutable receipt only after complete ingestion."""
    import numpy as np
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    catalog = load_catalog(documents)
    import csv
    with Path(catalog_csv).open(encoding="utf-8", newline="") as source:
        ids = [int(r["movie_id"]) for r in csv.DictReader(source)]
    if len(ids) != len(set(ids)) or set(ids) != set(catalog):
        raise ValueError("Catalog CSV and document IDs disagree")
    ordered = sorted(catalog)
    vectors = np.asarray(encoder.encode([catalog[i]["text"] for i in ordered]), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(ordered) or not np.isfinite(vectors).all():
        raise ValueError("Invalid document embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1, atol=1e-5):
        raise ValueError("Embeddings must be normalized and nonzero")
    identity = {
        "schema_version": "online-chroma-index-v1", "documents_sha256": file_hash(documents),
        "catalog_sha256": file_hash(catalog_csv), "embedding": encoder.settings,
        "vectors_sha256": hashlib.sha256(vectors.astype("<f4").tobytes()).hexdigest(),
        "dimension": vectors.shape[1], "count": len(ordered), "distance": "cosine",
        "chroma_version": importlib.metadata.version("chromadb"),
        "index_settings": {"space": "cosine", "ef_search": 100, "num_threads": 1},
    }
    version = "movies-" + hashlib.sha256(canonical(identity)).hexdigest()[:32]
    collection = client.create_collection(name=version, embedding_function=None,
        configuration={"hnsw": identity["index_settings"]}, metadata={"ready": False})
    published = False
    try:
        for start in range(0, len(ordered), 256):
            batch = ordered[start:start + 256]
            collection.add(ids=[str(i) for i in batch],
                           embeddings=vectors[start:start + len(batch)].tolist(),
                           metadatas=[{"movie_id": i} for i in batch])
        if collection.count() != len(ordered):
            raise ValueError("Incomplete index")
        receipt = {**identity, "index_version": version}
        collection.modify(metadata={"ready": True, "receipt_sha256": hashlib.sha256(canonical(receipt)).hexdigest()})
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("xb") as out:
            published = True
            out.write(canonical(receipt))
        return receipt
    except BaseException:
        if published:
            manifest_path.unlink(missing_ok=True)
        client.delete_collection(version)
        raise


class OnlineRetriever:
    def __init__(self, client, manifest_path, documents, encoder, pool_size=50):
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.index_version = self.manifest["index_version"]
        self.catalog = load_catalog(documents)
        self.encoder, self.pool_size = encoder, pool_size
        if file_hash(documents) != self.manifest["documents_sha256"]:
            raise ValueError("Index/catalog document hash mismatch")
        if encoder.settings != self.manifest["embedding"]:
            raise ValueError("Embedding configuration does not match index")
        self.collection = client.get_collection(self.index_version, embedding_function=None)
        if (self.collection.count() != len(self.catalog) or len(self.catalog) != self.manifest["count"]
                or not self.collection.metadata.get("ready")
                or self.collection.metadata.get("receipt_sha256") != hashlib.sha256(canonical(self.manifest)).hexdigest()):
            raise ValueError("Index receipt or completeness mismatch")

    def check_health(self):
        try:
            return self.collection.count() == self.manifest["count"]
        except Exception:
            return False

    def initial_pool(self, *, profile_text, clarifications, history_movie_ids):
        # MiniLM truncates at 256 tokens. Put the latest answer first so a long
        # canonical profile cannot truncate away the user's newly stated preference.
        updates = "\n".join(f"Answer: {t['answer']}\nQuestion: {t['question']}" for t in reversed(clarifications))
        query = updates + "\n" + profile_text if updates else profile_text
        if not query.strip():
            query = "Movie recommendations"
        return self.search(query=query, exclude_movie_ids=history_movie_ids, max_new=self.pool_size)

    def search(self, *, query, exclude_movie_ids, max_new):
        import numpy as np
        if not isinstance(query, str) or not query.strip() or type(max_new) is not int or max_new <= 0:
            raise ValueError("Search requires nonempty query and positive max_new")
        if any(type(i) is not int or i <= 0 for i in exclude_movie_ids):
            raise ValueError("Excluded IDs must be positive integers")
        excluded = set(exclude_movie_ids)
        count = min(max_new, len(set(self.catalog) - excluded))
        if not count:
            return RetrievalResult([], self.index_version)
        vector = np.asarray(self.encoder.encode([query]), dtype=np.float32)
        if (vector.shape != (1, self.manifest["dimension"]) or not np.isfinite(vector).all()
                or not np.allclose(np.linalg.norm(vector, axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid query embedding")
        kwargs = {"where": {"movie_id": {"$nin": sorted(excluded)}}} if excluded else {}
        result = self.collection.query(query_embeddings=vector.tolist(), n_results=count,
                                       include=["distances"], **kwargs)
        pairs = [(int(mid), float(d)) for mid, d in zip(result["ids"][0], result["distances"][0])]
        if (len({mid for mid, _ in pairs}) != len(pairs) or any(
                mid not in self.catalog or mid in excluded or not math.isfinite(d) for mid, d in pairs)):
            raise ValueError("Index returned invalid candidates")
        pairs.sort(key=lambda p: (p[1], p[0]))
        return RetrievalResult([mid for mid, _ in pairs], self.index_version)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", default="data/processed/ml25m-global-v1/movie_documents.jsonl")
    p.add_argument("--catalog", default="data/processed/ml25m-global-v1/catalog.csv")
    p.add_argument("--manifest", required=True)
    p.add_argument("--chroma-path", default="data/indexes/chroma")
    p.add_argument("--chroma-host")
    p.add_argument("--chroma-port", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print(json.dumps(ingest(chroma_client(args.chroma_path, args.chroma_host, args.chroma_port),
          args.documents, args.catalog, args.manifest, SentenceEncoder(device=args.device)), indent=2))


if __name__ == "__main__":
    main()
