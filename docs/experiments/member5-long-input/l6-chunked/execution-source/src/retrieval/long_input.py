"""Validation experiments preserving canonical text while varying its encoding."""
from __future__ import annotations

import numpy as np

from src.retrieval.embedding import normalized
from src.retrieval.run_embedding import SentenceEncoder


def token_chunks(ids: list[int], capacity: int) -> list[list[int]]:
    if type(capacity) is not int or capacity <= 0 or not ids:
        raise ValueError('Chunk capacity and content must be nonempty/positive')
    return [ids[start:start + capacity] for start in range(0, len(ids), capacity)]


def weighted_pool(vectors, owners, weights, count: int) -> np.ndarray:
    vectors = normalized(vectors)
    owners = np.asarray(owners)
    weights = np.asarray(weights, dtype=np.float32)
    if (type(count) is not int or count <= 0 or owners.shape != (len(vectors),)
            or owners.dtype.kind not in 'iu' or (owners < 0).any() or (owners >= count).any()
            or weights.shape != (len(vectors),) or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError('Invalid chunk ownership or weights')
    sums = np.zeros((count, vectors.shape[1]), dtype=np.float32)
    totals = np.zeros(count, dtype=np.float32)
    np.add.at(sums, owners, vectors * weights[:, None])
    np.add.at(totals, owners, weights)
    if (totals == 0).any():
        raise ValueError('Every input must have at least one chunk')
    return normalized(sums / totals[:, None])


def encoding_policy(config, kind):
    if kind not in ('movies', 'profiles'):
        raise ValueError('Unknown encoding role')
    strategy = config['encoding_strategy']
    if strategy not in ('chunked', 'full'):
        raise ValueError('Unknown encoding strategy')
    if strategy == 'chunked':
        if config.get('query_prompt'):
            raise ValueError('Chunking experiment does not use query instructions')
        return {'strategy': 'truncate' if kind == 'movies' else 'chunked', 'prompt': ''}
    prompt = config.get('query_prompt', '') if kind == 'profiles' else ''
    if not isinstance(prompt, str):
        raise ValueError('query_prompt must be a string')
    return {'strategy': 'full', 'prompt': prompt}


def policy_identity(config, kind):
    policy = encoding_policy(config, kind)
    if policy['strategy'] == 'truncate':
        return {}  # L6 movies are exactly the original baseline encoding/cache.
    return {'long_input_policy': {**policy, 'version': 1,
            'aggregation': 'content_token_weighted_normalized_mean' if policy['strategy'] == 'chunked' else 'model_pooling',
            'padding_side': 'right' if policy['strategy'] == 'chunked' else 'left'}}


def check_full_lengths(lengths, limit):
    if not lengths or max(lengths) > limit:
        raise ValueError(f'Full-input encoding would truncate: max={max(lengths, default=0)}, limit={limit}')


class LongInputEncoder(SentenceEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model.eval()
        self.model.tokenizer.padding_side = 'left' if config['encoding_strategy'] == 'full' else 'right'
        self._profile_tokens = None

    def cache_identity(self, kind):
        return policy_identity(self.config, kind)

    def describe(self, texts, kind):
        policy = encoding_policy(self.config, kind)
        tokenizer = self.model.tokenizer
        lengths, content_tokens = [], []
        for start in range(0, len(texts), 256):
            batch = [policy['prompt'] + text for text in texts[start:start + 256]]
            encoded = tokenizer(batch, add_special_tokens=policy['strategy'] != 'chunked',
                                truncation=False, padding=False, verbose=False)['input_ids']
            if policy['strategy'] == 'chunked':
                content_tokens.extend(encoded)
                lengths.extend(len(ids) + tokenizer.num_special_tokens_to_add(pair=False) for ids in encoded)
            else:
                lengths.extend(map(len, encoded))
        limit = self.config['max_seq_length']
        if policy['strategy'] == 'full':
            check_full_lengths(lengths, limit)
        summary = {'records': len(texts), 'max_tokens_before_truncation': max(lengths),
                   'token_limit': limit, 'over_limit_before_strategy': sum(n > limit for n in lengths),
                   'truncated': sum(n > limit for n in lengths) if policy['strategy'] == 'truncate' else 0,
                   'strategy': policy['strategy'], 'includes_query_prompt': bool(policy['prompt'])}
        if policy['strategy'] == 'chunked':
            self._profile_tokens = content_tokens
            capacity = limit - tokenizer.num_special_tokens_to_add(pair=False)
            chunks = [token_chunks(ids, capacity) for ids in content_tokens]
            summary.update({'chunks': sum(map(len, chunks)), 'max_chunks_per_record': max(map(len, chunks)),
                            'content_tokens': sum(map(len, content_tokens)), 'dropped_content_tokens': 0})
        return summary

    def encode_kind(self, texts, kind):
        policy = encoding_policy(self.config, kind)
        # Public callers get the same no-truncation guard as the runner, even without describe().
        self.describe(texts, kind)
        if policy['strategy'] != 'chunked':
            return self.model.encode(texts, prompt=policy['prompt'], batch_size=self.batch_size,
                                     show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        import torch
        from tqdm.auto import tqdm
        tokenizer = self.model.tokenizer
        capacity = self.config['max_seq_length'] - tokenizer.num_special_tokens_to_add(pair=False)
        if (tokenizer.cls_token_id is None or tokenizer.sep_token_id is None
                or tokenizer.num_special_tokens_to_add(pair=False) != 2):
            raise ValueError('Chunk experiment requires BERT CLS/content/SEP input format')
        features, owners, weights = [], [], []
        for owner, ids in enumerate(self._profile_tokens):
            for chunk in token_chunks(ids, capacity):
                # Pinned MiniLM uses BERT input format. Avoid decode/re-encode, which can
                # change subword boundaries, and removed Transformers prepare_for_model APIs.
                input_ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
                features.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids),
                                 'token_type_ids': [0] * len(input_ids)})
                owners.append(owner)
                weights.append(len(chunk))
        vectors = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(features), self.batch_size), desc='Encoding profile chunks'):
                batch = tokenizer.pad(features[start:start + self.batch_size], padding=True, return_tensors='pt')
                batch = {key: value.to(self.model.device) for key, value in batch.items()}
                embedding = self.model(batch)['sentence_embedding']
                vectors.append(embedding.detach().float().cpu().numpy())
        return weighted_pool(np.concatenate(vectors), owners, weights, len(texts))
