"""Exact cosine retrieval and content-addressed encoding caches (Member 5)."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from filelock import FileLock


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False, separators=(',', ':')) + '\n')
    temp.replace(path)


def validate_records(records: list[dict], id_field: str) -> list[dict]:
    if not records:
        raise ValueError('Empty input records')
    ids = set()
    for record in records:
        item_id = record.get(id_field)
        if type(item_id) is not int or item_id <= 0 or item_id in ids:
            raise ValueError(f'Invalid or duplicate {id_field}: {item_id}')
        ids.add(item_id)
        if not isinstance(record.get('text'), str) or not record['text'].strip():
            raise ValueError(f'Missing canonical text for {item_id}')
        if id_field == 'user_id':
            history = record.get('history_movie_ids')
            if (not isinstance(history, list) or any(type(i) is not int or i <= 0 for i in history)
                    or len(history) != len(set(history))):
                raise ValueError(f'Invalid full history for user {item_id}')
    return sorted(records, key=lambda r: r[id_field])


def normalized(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError('Vectors must be a nonempty finite matrix')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError('Zero or nonfinite vector norm')
    return np.ascontiguousarray(array / norms)


class ExactIndex:
    """Normalized matrix + aligned movie IDs; scores descend, ties use ID ascending."""
    def __init__(self, movie_ids, vectors):
        ids = np.asarray(movie_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0
                or (ids <= 0).any() or len(np.unique(ids)) != len(ids)):
            raise ValueError('Movie IDs must be unique positive integers')
        self.vectors = normalized(vectors)
        if len(ids) != len(self.vectors):
            raise ValueError('Vector/ID length mismatch')
        self.ids = ids.astype(np.int64)
        self.positions = {int(movie_id): i for i, movie_id in enumerate(self.ids)}

    @classmethod
    def load(cls, path: Path):
        """Validate a persisted index without a second floating-point normalization."""
        with np.load(path, allow_pickle=False) as saved:
            ids, vectors = saved['movie_ids'], saved['vectors']
        index = cls(ids, vectors)
        if vectors.dtype != np.float32 or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6, atol=1e-7):
            raise ValueError('Persisted index must contain normalized float32 vectors')
        index.vectors = np.ascontiguousarray(vectors)
        return index

    def search(self, query, watched: set[int], k: int) -> list[dict]:
        if type(k) is not int or k <= 0:
            raise ValueError('k must be a positive integer')
        query = normalized([query])[0]
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError('Query dimension mismatch')
        scores = self.vectors @ query
        seen = [self.positions[movie_id] for movie_id in watched if movie_id in self.positions]
        scores[seen] = -np.inf
        if np.isfinite(scores).sum() < k:
            raise ValueError(f'Fewer than {k} unseen catalog movies')
        order = np.lexsort((self.ids, -scores))[:k]
        return [{'movie_id': int(self.ids[i]), 'score': float(scores[i]), 'rank': rank}
                for rank, i in enumerate(order, 1)]


def cached_encode(path: Path, key: str, texts: list[str], encode) -> tuple[np.ndarray, bool]:
    """Serialize readers/builders for one cache entry, including both published files."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file after release: unlinking it can create two lock identities.
    with FileLock(str(path) + '.lock', timeout=600):
        return _cached_encode_locked(path, key, texts, encode)


def _cached_encode_locked(path: Path, key: str, texts: list[str], encode) -> tuple[np.ndarray, bool]:
    """Different input key rebuilds; matching corrupt cache fails visibly."""
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta['key'] == key:
                if meta['sha256'] != file_hash(path):
                    raise ValueError('Embedding cache checksum mismatch')
                with np.load(path, allow_pickle=False) as saved:
                    vectors = saved['vectors']
                if len(vectors) != len(texts) or not np.isfinite(vectors).all():
                    raise ValueError('Embedding cache shape/content mismatch')
                # Validate, but preserve bytes so cached and uncached search agree.
                normalized(vectors)
                return vectors, True
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid cache {path}') from exc
    vectors = normalized(encode(texts))
    if len(vectors) != len(texts):
        raise ValueError('Encoder returned wrong number of vectors')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as out:
        temp = Path(out.name)
        try:
            np.savez(out, vectors=vectors)
        except BaseException:
            out.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
        write_json(meta_path, {'key': key, 'sha256': file_hash(path), 'shape': list(vectors.shape)})
    finally:
        temp.unlink(missing_ok=True)
    return vectors, False
