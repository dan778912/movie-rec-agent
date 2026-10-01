"""Reproducible B3 embedding retrieval and fixed candidate export."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import shutil
import tempfile
import sys
import time
from pathlib import Path


def check_python_version(version):
    if tuple(version[:2]) < (3, 12):
        raise RuntimeError('Member 5 locked dependencies require Python 3.12 or newer; use Python 3.12.')


check_python_version(sys.version_info)

import numpy as np
from filelock import FileLock, Timeout
from threadpoolctl import threadpool_limits

from src.retrieval.embedding import (ExactIndex, cached_encode, file_hash, fingerprint,
                                     read_jsonl, validate_records, write_json, write_jsonl)
from src.retrieval.diagnostics import validate_predictions


class SentenceEncoder:
    def __init__(self, config):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(config['seed'])
        torch.set_num_threads(config.get('threads', 4))
        self.model = SentenceTransformer(config['model_id'], revision=config['revision'],
                                         device=config['device'], trust_remote_code=False)
        self.model.max_seq_length = config['max_seq_length']
        self.batch_size = config['batch_size']

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)

    def token_lengths(self, texts):
        lengths = []
        for start in range(0, len(texts), 256):
            encoded = self.model.tokenizer(texts[start:start + 256], truncation=False,
                                            padding=False, verbose=False)
            lengths.extend(len(ids) for ids in encoded['input_ids'])
        return lengths


def runtime_versions():
    names = ('numpy', 'torch', 'sentence-transformers', 'transformers', 'tokenizers')
    return {name: importlib.metadata.version(name) for name in names}


def code_identity():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'src' / 'retrieval').glob('*.py'))
    files += [root / 'requirements-embedding.txt'] if (root / 'requirements-embedding.txt').exists() else []
    hashes = {str(p.relative_to(root)): file_hash(p) for p in files}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'git_commit': commit, 'working_tree_dirty': dirty,
            'source_sha256': hashes, 'source_fingerprint': fingerprint(hashes)}


def validate_config(config):
    if not re.fullmatch(r'[0-9a-f]{40}', config.get('revision', '')):
        raise ValueError('Pin model to a full immutable 40-character revision')
    for field in ('max_seq_length', 'batch_size', 'top_k', 'pool_size'):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f'{field} must be a positive integer')
    if config['pool_size'] > config['top_k']:
        raise ValueError('pool_size must be <= top_k')


def _run_staged(config, data: Path, output: Path, cache: Path, split: str, encoder=None):
    started = time.perf_counter()
    validate_config(config)
    if split not in ('validation', 'test'):
        raise ValueError('Invalid split')
    # All runs have a complete manifest only on success; no silent overwrite of frozen output.
    if (output / 'run.json').exists():
        raise ValueError('Output already completed; use a new output directory')
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = {}
    for name in ('movie_documents.jsonl', f'{split}_profiles.jsonl'):
        actual = file_hash(data / name)
        if actual != manifest['file_sha256'].get(name):
            raise ValueError(f'Data hash does not match frozen manifest: {name}')
        inputs[name] = actual
    docs = validate_records(read_jsonl(data / 'movie_documents.jsonl'), 'movie_id')
    profiles = validate_records(read_jsonl(data / f'{split}_profiles.jsonl'), 'user_id')
    if any(p.get('split') != split for p in profiles):
        raise ValueError('Profile split mismatch')
    versions = runtime_versions()
    start = time.perf_counter()
    encoder = encoder or SentenceEncoder(config)
    model_load_seconds = time.perf_counter() - start
    identity = {key: config[key] for key in ('model_id', 'revision', 'max_seq_length', 'device')}
    identity.update({'runtime': versions, 'normalization': 'l2_float32_v1'})
    vectors = {}
    timings = {}
    hits = {}
    truncation = {}
    cache_files = {}
    for kind, records, filename in [('movies', docs, 'movie_documents.jsonl'),
                                     ('profiles', profiles, f'{split}_profiles.jsonl')]:
        texts = [r['text'] for r in records]
        key = fingerprint({'encoder': identity, 'input_sha256': inputs[filename]})
        path = cache / f'{kind}-{key}.npz'
        start = time.perf_counter()
        vectors[kind], hits[kind] = cached_encode(path, key, texts, encoder.encode)
        timings[kind + '_encode_or_cache_seconds'] = time.perf_counter() - start
        cache_files[kind] = str(path.resolve())
        lengths = encoder.token_lengths(texts)
        truncation[kind] = {'records': len(lengths),
                            'truncated': sum(n > config['max_seq_length'] for n in lengths),
                            'max_tokens_before_truncation': max(lengths),
                            'token_limit': config['max_seq_length']}
    index = ExactIndex([d['movie_id'] for d in docs], vectors['movies'])
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / 'index.npz'
    with index_path.open('wb') as out:
        np.savez(out, movie_ids=index.ids, vectors=index.vectors)
    predictions, pools, durations = [], [], []
    with threadpool_limits(limits=1):
        for profile, query in zip(profiles, vectors['profiles']):
            start = time.perf_counter()
            recommendations = index.search(query, set(profile['history_movie_ids']), config['top_k'])
            durations.append(time.perf_counter() - start)
            row = {'user_id': profile['user_id'], 'split': split, 'method': 'embedding',
                   'recommendations': recommendations}
            predictions.append(row)
            pools.append({**row, 'recommendations': recommendations[:config['pool_size']]})
    histories = {p['user_id']: set(p['history_movie_ids']) for p in profiles}
    catalog = set(int(i) for i in index.ids)
    validate_predictions(predictions, catalog, histories, k=config['top_k'], split=split)
    validate_predictions(pools, catalog, histories, k=config['pool_size'], split=split)
    # File name remains the established handoff name; config records count for small fixtures.
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'candidates_top50.jsonl', pools)
    hashes = {name: file_hash(output / name) for name in
              ('predictions.jsonl', 'candidates_top50.jsonl', 'index.npz')}
    report = {'config': config, 'split': split, 'users': len(profiles), 'movies': len(docs),
              'data_version': manifest['version'], 'manifest_sha256': file_hash(manifest_path),
              'input_sha256': inputs, 'output_sha256': hashes, 'cache_files': cache_files,
              'cache_hits': hits, 'truncation': truncation, 'runtime': versions,
              'platform': platform.platform(), 'python': platform.python_version(),
              'code': code_identity(), 'error_count': 0, 'index': 'exact_numpy_cosine_float32',
              'timings': {**timings, 'model_load_seconds': model_load_seconds,
                          'retrieval_total_seconds': sum(durations),
                          'retrieval_mean_ms': float(np.mean(durations) * 1000),
                          'retrieval_p50_ms': float(np.percentile(durations, 50) * 1000),
                          'retrieval_p95_ms': float(np.percentile(durations, 95) * 1000),
                          'total_seconds': time.perf_counter() - started},
              'timing_note': 'Retrieval latency excludes query encoding; separate batch encoding times are reported.',
              'training_seconds': 0, 'api_inference_cost_usd': 0}
    write_json(output / 'run.json', report)
    return report


def run(config, data: Path, output: Path, cache: Path, split: str, encoder=None):
    """Reserve an output name before work, then atomically publish a complete directory."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output.parent / ('.' + output.name + '.lock')), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise ValueError(f'Output is in use by another run: {output}') from exc
    try:
        # Refuse even partial/pre-existing directories; never remove another run's files.
        if output.exists():
            raise ValueError('Output already exists; use a new output directory')
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.staging-', dir=output.parent))
        try:
            report = _run_staged(config, data, staging, cache, split, encoder)
            if output.exists():
                raise ValueError('Output appeared during run; refusing to replace it')
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        lock.release()
    print(json.dumps({'output': str(output), 'users': report['users'],
                      'cache_hits': report['cache_hits'], 'timings': report['timings']}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.data, args.output, args.cache, args.split)


if __name__ == '__main__':
    main()
