"""Provisional Member 5 validation diagnostics, NOT the authoritative team evaluator."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from src.retrieval.embedding import read_jsonl, write_json


def validate_predictions(predictions, catalog, histories, k=None, split=None):
    seen_users = set()
    for row in predictions:
        user = row['user_id']
        if user in seen_users or user not in histories:
            raise ValueError(f'Duplicate or unexpected user {user}')
        seen_users.add(user)
        if split is not None and row.get('split') != split:
            raise ValueError('Prediction split mismatch')
        recs = row['recommendations']
        if k is not None and len(recs) != k:
            raise ValueError('Wrong candidate count')
        ids = [r['movie_id'] for r in recs]
        if len(set(ids)) != len(ids) or not set(ids) <= catalog or set(ids) & histories[user]:
            raise ValueError('Duplicate, out-of-catalog, or watched recommendation')
        if [r['rank'] for r in recs] != list(range(1, len(recs) + 1)):
            raise ValueError('Ranks must be consecutive starting at 1')
        if any(not math.isfinite(r['score']) for r in recs):
            raise ValueError('Nonfinite score')
    if seen_users != set(histories):
        raise ValueError('Missing evaluation users')


def evaluate(predictions, targets, catalog, histories, ks=(10, 20, 50, 100)):
    validate_predictions(predictions, catalog, histories)
    if not histories:
        raise ValueError('No evaluation users')
    positives = {user: set() for user in histories}
    total = in_catalog = 0
    for row in targets:
        user, movie = int(row['user_id']), int(row['movie_id'])
        flag = int(row['in_catalog'])
        if flag not in (0, 1) or bool(flag) != (movie in catalog):
            raise ValueError('Target catalog flag inconsistent')
        # Coverage is across the full target file (including users with only excluded targets).
        total += 1
        in_catalog += flag
        if user in positives and flag:
            positives[user].add(movie)
    if any(not ids for ids in positives.values()):
        raise ValueError('Evaluation user has no in-catalog positives')
    metrics = {f'{metric}@{k}': 0.0 for metric in ('recall', 'ndcg') for k in ks}
    for row in predictions:
        relevant = positives[row['user_id']]
        recommendations = row['recommendations']
        if len(recommendations) < max(ks):
            raise ValueError('Not enough recommendations for requested metrics')
        for k in ks:
            hits = [int(r['movie_id'] in relevant) for r in recommendations[:k]]
            metrics[f'recall@{k}'] += sum(hits) / len(relevant)
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
            metrics[f'ndcg@{k}'] += dcg / idcg
    return {**{key: value / len(histories) for key, value in metrics.items()},
            'status': 'provisional_member5_diagnostic_not_team_evaluator',
            'users': len(histories), 'positive_targets_all_users': total,
            'excluded_positive_targets': total - in_catalog,
            'catalog_positive_coverage': in_catalog / total if total else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Intentionally validation-only: no accidental final-test tuning.
    profiles = read_jsonl(args.data / 'validation_profiles.jsonl')
    docs = read_jsonl(args.data / 'movie_documents.jsonl')
    predictions = read_jsonl(args.predictions)
    histories = {p['user_id']: set(p['history_movie_ids']) for p in profiles}
    catalog = {d['movie_id'] for d in docs}
    validate_predictions(predictions, catalog, histories, k=100, split='validation')
    with (args.data / 'validation_targets.csv').open() as source:
        result = evaluate(predictions, list(csv.DictReader(source)), catalog, histories)
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
