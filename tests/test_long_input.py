"""Behavioral tests for the two validation-only long-input experiments."""
import unittest
import numpy as np
from src.retrieval import run_embedding


class LongInputTests(unittest.TestCase):
    def module(self):
        # Start with an explicit assertion so the red test is a missing feature, not an import typo.
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('src.retrieval.long_input'))
        from src.retrieval import long_input
        return long_input

    def test_chunks_preserve_tail_and_every_token_once(self):
        split = self.module().token_chunks
        self.assertEqual(split([11, 12, 13, 14, 15], 4), [[11, 12, 13, 14], [15]])
        self.assertEqual(split([11, 12, 13, 14], 4), [[11, 12, 13, 14]])
        for ids, size in [([], 4), ([1], 0), ([1], True)]:
            with self.assertRaises(ValueError):
                split(ids, size)

    def test_weighted_pool_uses_content_count_not_equal_chunk_weights(self):
        pool = self.module().weighted_pool
        # First user's orthogonal chunks have lengths 3 and 1 -> direction (3,1).
        result = pool(np.array([[1., 0.], [0., 1.], [0., 2.]]), [0, 0, 1], [3, 1, 2], 2)
        np.testing.assert_allclose(result, [[3/np.sqrt(10), 1/np.sqrt(10)], [0, 1]], rtol=1e-6)
        with self.assertRaises(ValueError):
            pool(np.array([[1., 0.]]), [0], [0], 1)
        with self.assertRaises(ValueError):
            pool(np.array([[1., 0.]]), [0], [1], 2)

    def test_prompt_applies_only_to_profiles_and_full_text_overflow_fails(self):
        prepare = self.module().encoding_policy
        cfg = {'encoding_strategy': 'full', 'query_prompt': 'Instruct: recommend\nQuery: '}
        self.assertEqual(prepare(cfg, 'movies'), {'strategy': 'full', 'prompt': ''})
        self.assertEqual(prepare(cfg, 'profiles'), {'strategy': 'full', 'prompt': 'Instruct: recommend\nQuery: '})
        self.module().check_full_lengths([3, 10], 10)
        with self.assertRaises(ValueError):
            self.module().check_full_lengths([3, 11], 10)

    def test_strategy_and_prompt_separate_cache_identity_but_l6_movies_reuse(self):
        identity = self.module().policy_identity
        self.assertEqual(identity({'encoding_strategy': 'chunked'}, 'movies'), {})
        chunk = identity({'encoding_strategy': 'chunked'}, 'profiles')
        full = identity({'encoding_strategy': 'full', 'query_prompt': 'one'}, 'profiles')
        other = identity({'encoding_strategy': 'full', 'query_prompt': 'two'}, 'profiles')
        self.assertNotEqual(chunk, full)
        self.assertNotEqual(full, other)
        half = identity({'encoding_strategy': 'full', 'query_prompt': 'one', 'model_dtype':'float16'}, 'profiles')
        self.assertNotEqual(full, half)
        with self.assertRaises(ValueError):
            identity({'encoding_strategy': 'unknown'}, 'profiles')

    def test_new_experiments_cannot_run_on_test_split(self):
        # Guard must fail before reading files or loading an encoder.
        from pathlib import Path
        cfg = {'model_id': 'fixture', 'revision': 'a'*40, 'max_seq_length':256,
               'batch_size': 2, 'seed':42, 'top_k':100, 'pool_size':50, 'device':'cpu',
               'encoding_strategy':'chunked'}
        with self.assertRaisesRegex(ValueError, 'validation'):
            run_embedding._run_staged(cfg, Path('/does-not-exist'), Path('/does-not-exist'),
                                      Path('/does-not-exist'), 'test')

    def test_real_bert_tokenizer_chunk_forward_preserves_final_token(self):
        import torch
        from transformers import BertTokenizer
        from src.retrieval.long_input import LongInputEncoder
        tokenizer = BertTokenizer(vocab={'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,
                                         '[MASK]':4,'a':5,'b':6})
        class TinyEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer, self.device = tokenizer, torch.device('cpu')
            def forward(self, features):
                # Real tokenization and batching; a deterministic stand-in for costly weights.
                ids = features['input_ids']
                return {'sentence_embedding': torch.stack(((ids == 5).sum(1), (ids == 6).sum(1)),dim=1).float()}
        encoder = LongInputEncoder.__new__(LongInputEncoder)
        encoder.model, encoder.batch_size = TinyEmbedding(), 2
        encoder.config = {'encoding_strategy':'chunked','max_seq_length':6}
        encoder._profile_tokens = None
        result = encoder.encode_kind(['a a a a b'], 'profiles')
        # Four a's in chunk one; final b must survive as a length-one second chunk.
        np.testing.assert_allclose(result, [[4/np.sqrt(17),1/np.sqrt(17)]],rtol=1e-6)
        stats = encoder.describe(['a a a a b'], 'profiles')
        self.assertEqual(stats['chunks'],2)
        self.assertEqual(stats['dropped_content_tokens'],0)

    def test_full_encoder_routes_prompt_and_rejects_overflow_before_inference(self):
        from transformers import BertTokenizer
        from src.retrieval.long_input import LongInputEncoder
        tokenizer = BertTokenizer(vocab={'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,
                                         '[MASK]':4,'a':5,'b':6})
        class SmallModel:
            def __init__(self):
                self.tokenizer, self.calls = tokenizer, []
            def encode(self, texts, **kwargs):
                self.calls.append((texts, kwargs['prompt']))
                return np.array([[len(kwargs['prompt']), len(text)] for text in texts])
        encoder = LongInputEncoder.__new__(LongInputEncoder)
        encoder.model, encoder.batch_size = SmallModel(), 2
        encoder.config = {'encoding_strategy':'full','max_seq_length':5,'query_prompt':'a a '}
        np.testing.assert_array_equal(encoder.encode_kind(['b'], 'movies'), [[0,1]])
        np.testing.assert_array_equal(encoder.encode_kind(['b'], 'profiles'), [[4,1]])
        with self.assertRaises(ValueError):
            encoder.encode_kind(['b b b b'], 'profiles')
        self.assertEqual(len(encoder.model.calls),2)


if __name__ == '__main__':
    unittest.main()
