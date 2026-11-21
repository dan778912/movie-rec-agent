"""Regression coverage for the external Member 5 review."""
import copy
import json
import multiprocessing
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.retrieval.embedding import cached_encode, file_hash, write_json, write_jsonl
from src.retrieval.run_embedding import run
from src.retrieval.diagnostics import evaluate, validate_predictions


class SmallEncoder:
    def encode(self, texts):
        return np.array([[len(t), 1] for t in texts], dtype=np.float32)

    def token_lengths(self, texts):
        return [len(t.split()) for t in texts]


def fixture(root):
    write_jsonl(root / 'movie_documents.jsonl',
                [{'movie_id': i, 'text': f'movie {i}'} for i in range(1, 7)])
    for split in ('validation', 'test'):
        write_jsonl(root / f'{split}_profiles.jsonl',
                    [{'user_id': 1, 'text': 'movie fan', 'split': split, 'history_movie_ids': [1]}])
    write_json(root / 'manifest.json', {'version': 'fixture', 'file_sha256':
               {p.name: file_hash(p) for p in root.glob('*.jsonl')}})
    return {'model_id': 'fixture', 'revision': 'a'*40, 'max_seq_length': 256,
            'batch_size': 2, 'seed': 42, 'top_k': 4, 'pool_size': 2, 'device': 'cpu'}


def cache_worker(path, entered, release, attempting, results):
    def encode(texts):
        entered.set()
        if not release.wait(10):
            raise TimeoutError('Test synchronization timed out')
        return np.array([[1, 2]], dtype=np.float32)
    attempting.set()
    try:
        vectors, hit = cached_encode(Path(path), 'same-key', ['movie'], encode)
        results.put(('ok', hit, vectors.tolist()))
    except Exception as exc:
        results.put(('error', type(exc).__name__, str(exc)))


class ConcurrencyTests(unittest.TestCase):
    def test_same_output_contender_rejected_and_only_complete_directory_published(self):
        entered, release = threading.Event(), threading.Event()
        class SlowEncoder(SmallEncoder):
            def encode(self, texts):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('Test synchronization timed out')
                return super().encode(texts)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root)
            out = root / 'shared-output'
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(run, config, root, out, root/'cache-a', 'validation', SlowEncoder())
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertFalse(out.exists(), 'No partially published output should be visible')
                    with self.assertRaises(ValueError):
                        run(config, root, out, root/'cache-b', 'test', SmallEncoder())
                finally:
                    release.set()
                    first.result(timeout=10)
            self.assertEqual(json.loads((out/'run.json').read_text())['split'], 'validation')
            original = file_hash(out/'predictions.jsonl')
            with self.assertRaises(ValueError):
                run(config, root, out, root/'cache-b', 'test', SmallEncoder())
            self.assertEqual(file_hash(out/'predictions.jsonl'), original)

    def test_failed_run_cleans_private_staging_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root)
            out = root/'output'
            with self.assertRaises(ValueError):
                run({**config, 'top_k': 10}, root, out, root/'cache', 'validation', SmallEncoder())
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob('.output.staging-*')), [])
            run(config, root, out, root/'cache', 'validation', SmallEncoder())
            self.assertTrue((out/'run.json').is_file())

    def test_shared_cache_cold_start_across_processes_encodes_once(self):
        ctx = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as temp:
            release = ctx.Event()
            first_entered, second_entered = ctx.Event(), ctx.Event()
            first_attempt, second_attempt = ctx.Event(), ctx.Event()
            results = ctx.Queue()
            path = str(Path(temp)/'vectors.npz')
            processes = [ctx.Process(target=cache_worker, args=(path, event, release, attempt, results))
                         for event, attempt in [(first_entered, first_attempt), (second_entered, second_attempt)]]
            processes[0].start()
            try:
                self.assertTrue(first_entered.wait(5))
                processes[1].start()
                self.assertTrue(second_attempt.wait(5))
                self.assertFalse(second_entered.wait(.5), 'Second writer must wait, then reuse published cache')
            finally:
                release.set()
                for process in processes:
                    if process.pid is not None:
                        process.join(10)
                        if process.is_alive():
                            process.terminate()
                            process.join()
            values = [results.get(timeout=2) for _ in range(2)]
            self.assertEqual([p.exitcode for p in processes], [0, 0])
            self.assertTrue(all(v[0] == 'ok' for v in values), values)
            self.assertEqual(sorted(v[1] for v in values), [False, True])
            self.assertEqual(values[0][2], values[1][2])
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])


class StrictDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.row = {'user_id': 1, 'split': 'validation', 'method': 'embedding',
                    'recommendations': [{'movie_id': 10, 'rank': 1, 'score': 1.0}]}
        self.targets = [{'user_id': 1, 'movie_id': 10, 'in_catalog': 1}]

    def test_duplicate_empty_nonpositive_noninteger_ks_rejected(self):
        for ks in [(1,1), (), (0,), (-1,), (True,), (1.0,), ('1',)]:
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                evaluate([self.row], self.targets, {10}, {1:set()}, ks=ks)
        self.assertEqual(evaluate([self.row], self.targets, {10}, {1:set()}, ks=(1,))['recall@1'], 1)

    def test_metadata_and_recommendation_types_are_checked(self):
        cases = []
        for key, value in [('user_id',True),('user_id',1.0),('method',''),('method',None),
                           ('split','train'),('split',False),('recommendations',{})]:
            row=copy.deepcopy(self.row); row[key]=value; cases.append(row)
        for key in ['user_id','method','split','recommendations']:
            row=copy.deepcopy(self.row); del row[key]; cases.append(row)
        for key,value in [('movie_id',10.0),('movie_id',True),('rank',True),('rank',1.0),
                          ('score',True),('score','1'),('score',float('nan'))]:
            row=copy.deepcopy(self.row); row['recommendations'][0][key]=value; cases.append(row)
        for key in ['movie_id','rank','score']:
            row=copy.deepcopy(self.row); del row['recommendations'][0][key]; cases.append(row)
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_predictions([row], {10}, {1:set()}, k=1)
        validate_predictions([self.row], {10}, {1:set()}, k=1, split='validation')


class EnvironmentTests(unittest.TestCase):
    def test_embedding_requires_python_312_or_newer(self):
        from src.retrieval.run_embedding import check_python_version
        for version in [(3, 10), (3, 11)]:
            with self.assertRaisesRegex(RuntimeError, '3.12'):
                check_python_version(version)
        check_python_version((3, 12))


if __name__ == '__main__':
    unittest.main()
