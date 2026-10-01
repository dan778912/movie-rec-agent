import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.embedding import ExactIndex, cached_encode, fingerprint, validate_records
from src.retrieval.diagnostics import evaluate


class RetrievalTests(unittest.TestCase):
    def test_cosine_order_ties_and_full_history_filter(self):
        index = ExactIndex([30, 10, 20, 40], [[1, 0], [1, 0], [0, 1], [-1, 0]])
        result = index.search([2, 0], {30, 999}, 3)
        self.assertEqual([r['movie_id'] for r in result], [10, 20, 40])
        self.assertEqual([r['rank'] for r in result], [1, 2, 3])
        np.testing.assert_allclose([r['score'] for r in result], [1, 0, -1])
        self.assertEqual(index.search([1, 0], set(), 2)[0]['movie_id'], 10)

    def test_reject_invalid_vectors_and_ids(self):
        for ids, vectors in [([1, 1], [[1, 0], [0, 1]]), ([1], [[0, 0]]),
                             ([1], [[np.nan, 1]]), ([1, 2], [[1, 0]]), ([], [])]:
            with self.subTest(ids=ids, vectors=vectors), self.assertRaises(ValueError):
                ExactIndex(ids, vectors)
        index = ExactIndex([1, 2], [[1, 0], [0, 1]])
        for query, seen, k in [([0, 0], set(), 1), ([1, 0, 0], set(), 1),
                               ([1, 0], {1, 2}, 1), ([1, 0], set(), 0)]:
            with self.subTest(query=query, k=k), self.assertRaises(ValueError):
                index.search(query, seen, k)

    def test_cache_reuses_only_matching_input_and_detects_corruption(self):
        calls = []
        def encode(texts):
            calls.append(texts)
            return np.array([[len(t), 1] for t in texts], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.npz'
            key = fingerprint({'model': 'abc', 'text': ['one', 'two']})
            first, hit = cached_encode(path, key, ['one', 'two'], encode)
            self.assertFalse(hit)
            again, hit = cached_encode(path, key, ['one', 'two'], encode)
            self.assertTrue(hit)
            np.testing.assert_array_equal(first, again)
            self.assertEqual(len(calls), 1)
            cached_encode(path, fingerprint({'model': 'def'}), ['one', 'two'], encode)
            self.assertEqual(len(calls), 2)
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                cached_encode(path, fingerprint({'model': 'def'}), ['one', 'two'], encode)

    def test_validate_duplicate_ids_empty_text_and_missing_history(self):
        self.assertEqual(validate_records([{'movie_id': 1, 'text': 'x'}], 'movie_id')[0]['movie_id'], 1)
        for rows in [[], [{'movie_id': 1, 'text': ''}],
                     [{'movie_id': 1, 'text': 'x'}, {'movie_id': 1, 'text': 'y'}]]:
            with self.assertRaises(ValueError):
                validate_records(rows, 'movie_id')
        with self.assertRaises(ValueError):
            validate_records([{'user_id': 1, 'text': 'x'}], 'user_id')


class DiagnosticTests(unittest.TestCase):
    def test_hand_calculated_macro_metrics_and_coverage(self):
        predictions = [
            {'user_id': 1, 'split': 'validation', 'method': 'embedding', 'recommendations':
             [{'movie_id': 10, 'score': .9, 'rank': 1}, {'movie_id': 30, 'score': .8, 'rank': 2}]},
            {'user_id': 2, 'split': 'validation', 'method': 'embedding', 'recommendations':
             [{'movie_id': 20, 'score': .9, 'rank': 1}, {'movie_id': 30, 'score': .8, 'rank': 2}]}]
        targets = [{'user_id': 1, 'movie_id': 10, 'in_catalog': 1},
                   {'user_id': 1, 'movie_id': 20, 'in_catalog': 1},
                   {'user_id': 1, 'movie_id': 99, 'in_catalog': 0},
                   {'user_id': 2, 'movie_id': 30, 'in_catalog': 1}]
        result = evaluate(predictions, targets, {10, 20, 30}, {1: {40}, 2: set()}, ks=(1, 2))
        self.assertAlmostEqual(result['recall@1'], .25)
        self.assertAlmostEqual(result['recall@2'], .75)
        expected = (1 / (1 + 1/math.log2(3)) + 1/math.log2(3)) / 2
        self.assertAlmostEqual(result['ndcg@2'], expected)
        self.assertEqual(result['excluded_positive_targets'], 1)
        self.assertEqual(result['catalog_positive_coverage'], .75)

    def test_invalid_predictions_fail_closed(self):
        base = {'user_id': 1, 'split': 'validation', 'method': 'embedding', 'recommendations':
                [{'movie_id': 10, 'rank': 1, 'score': 1.0}]}
        targets = [{'user_id': 1, 'movie_id': 10, 'in_catalog': 1}]
        for predictions, histories in [([base, base], {1: set()}), ([], {1: set()}),
                                        ([base], {1: {10}}), ([base], {2: set()})]:
            with self.assertRaises(ValueError):
                evaluate(predictions, targets, {10}, histories, ks=(1,))
        bad = dict(base, recommendations=base['recommendations'] * 2)
        with self.assertRaises(ValueError):
            evaluate([bad], targets, {10}, {1: set()}, ks=(1,))

if __name__ == '__main__':
    unittest.main()

class PipelineTests(unittest.TestCase):
    def test_pipeline_exports_matching_prefixes_and_reuses_cache(self):
        from src.retrieval.run_embedding import run
        from src.retrieval.embedding import file_hash, write_json, write_jsonl, read_jsonl
        class Encoder:
            calls = 0
            def encode(self, texts):
                self.calls += 1
                return np.array([[len(t), 1] for t in texts], dtype=np.float32)
            def token_lengths(self, texts):
                return [len(t.split()) + 2 for t in texts]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = [{'movie_id': i, 'text': 'movie ' + str(i)} for i in range(1, 7)]
            profiles = [{'user_id': 7, 'text': 'movie fan', 'split': 'validation', 'history_movie_ids': [1]}]
            write_jsonl(root / 'movie_documents.jsonl', docs)
            write_jsonl(root / 'validation_profiles.jsonl', profiles)
            hashes = {p.name: file_hash(p) for p in root.glob('*.jsonl')}
            write_json(root / 'manifest.json', {'version': 'fixture', 'file_sha256': hashes})
            config = {'model_id': 'fixture', 'revision': 'a'*40, 'max_seq_length': 256,
                      'batch_size': 2, 'seed': 42, 'top_k': 4, 'pool_size': 2, 'device': 'cpu'}
            encoder = Encoder()
            first = run(config, root, root / 'run1', root / 'cache', 'validation', encoder)
            second = run(config, root, root / 'run2', root / 'cache', 'validation', encoder)
            self.assertEqual(encoder.calls, 2)
            self.assertEqual(first['output_sha256'], second['output_sha256'])
            full = read_jsonl(root / 'run1' / 'predictions.jsonl')
            pool = read_jsonl(root / 'run1' / 'candidates_top50.jsonl')
            self.assertEqual(full[0]['recommendations'][:2], pool[0]['recommendations'])
            self.assertTrue(second['cache_hits']['movies'])
            self.assertEqual(second['users'], 1)
            # Changed source without matching Member 1 manifest must not be consumed.
            (root / 'movie_documents.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                run(config, root, root / 'run3', root / 'cache', 'validation', encoder)

class ReviewRegressionTests(unittest.TestCase):
    def test_index_reload_preserves_exact_scores_and_rank(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'index.npz'
            for _ in range(30):
                vectors = rng.normal(size=(2, 384)).astype(np.float32)
                index = ExactIndex([1, 2], vectors)
                query = index.vectors.sum(axis=0)
                with path.open('wb') as out:
                    np.savez(out, movie_ids=index.ids, vectors=index.vectors)
                restored = ExactIndex.load(path)
                np.testing.assert_array_equal(index.vectors, restored.vectors)
                self.assertEqual(index.search(query, set(), 2), restored.search(query, set(), 2))

    def test_diagnostic_rejects_changed_labels_and_records_provenance(self):
        from src.retrieval.diagnostics import diagnose
        from src.retrieval.embedding import file_hash, write_json, write_jsonl
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / 'movie_documents.jsonl', [{'movie_id': 1, 'text': 'a'}, {'movie_id': 2, 'text': 'b'}])
            write_jsonl(root / 'validation_profiles.jsonl',
                        [{'user_id': 1, 'text': 'a', 'split': 'validation', 'history_movie_ids': []}])
            targets = root / 'validation_targets.csv'
            targets.write_text('user_id,movie_id,in_catalog\n1,1,1\n')
            inputs = {p.name: file_hash(p) for p in root.iterdir()}
            write_json(root / 'manifest.json', {'version': 'fixture', 'file_sha256': inputs})
            prediction_path = root / 'predictions.jsonl'
            write_jsonl(prediction_path, [{'user_id': 1, 'split': 'validation', 'method': 'embedding',
                'recommendations': [{'movie_id': 1, 'rank': 1, 'score': 1.0}]}])
            result = diagnose(root, prediction_path, ks=(1,))
            self.assertEqual(result['input_sha256']['validation_targets.csv'], inputs['validation_targets.csv'])
            self.assertEqual(result['prediction_sha256'], file_hash(prediction_path))
            targets.write_text('user_id,movie_id,in_catalog\n1,2,1\n')
            with self.assertRaises(ValueError):
                diagnose(root, prediction_path, ks=(1,))
