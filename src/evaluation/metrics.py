"""Binary ranking metrics over MovieLens IDs, using only the standard library.

Each user's relevant IDs must include every distinct eligible held-out positive,
including positives absent from the retrieved candidate pool. Rank order is the
input order; scores and explanation text do not enter these calculations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real


def _movie_ids(values: Iterable[int], name: str, *, ordered: bool) -> list[int]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be an iterable of positive integer movie IDs")
    if ordered and isinstance(values, (set, frozenset)):
        raise TypeError(f"{name} must preserve recommendation order; sets are unordered")
    try:
        result = list(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of positive integer movie IDs") from error
    for index, movie_id in enumerate(result):
        if type(movie_id) is not int:
            raise TypeError(f"{name}[{index}] must be an integer movie ID, not {type(movie_id).__name__}")
        if movie_id <= 0:
            raise ValueError(f"{name}[{index}] must be a positive movie ID")
    return result


def _inputs(
    ranked_movie_ids: Iterable[int], relevant_movie_ids: Iterable[int], k: int
) -> tuple[list[int], set[int]]:
    if type(k) is not int:
        raise TypeError("k must be a positive integer; booleans are not valid cutoffs")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    ranked = _movie_ids(ranked_movie_ids, "ranked_movie_ids", ordered=True)
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_movie_ids contains duplicate movie IDs")
    relevant = set(_movie_ids(relevant_movie_ids, "relevant_movie_ids", ordered=False))
    if not relevant:
        raise ValueError("relevant_movie_ids must contain at least one eligible positive target")
    return ranked[:k], relevant


def recall_at_k(
    ranked_movie_ids: Iterable[int], relevant_movie_ids: Iterable[int], k: int
) -> float:
    """Return the fraction of all relevant targets found in the first k ranks.

    Duplicate rankings and empty truth are invalid. Empty/short rankings receive
    zero gain at missing positions. Relevant IDs are deduplicated as a set.
    Inputs are copied and may be lists, tuples, or generators; truth may be a set.
    """
    top_k, relevant = _inputs(ranked_movie_ids, relevant_movie_ids, k)
    return len(set(top_k) & relevant) / len(relevant)


def ndcg_at_k(
    ranked_movie_ids: Iterable[int], relevant_movie_ids: Iterable[int], k: int
) -> float:
    """Return binary NDCG with log2 discounts and full eligible-positive truth.

    Ideal gain uses min(k, number of relevant targets), independent of how many
    targets retrieval found or how many recommendations were returned.
    """
    top_k, relevant = _inputs(ranked_movie_ids, relevant_movie_ids, k)
    dcg = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank, movie_id in enumerate(top_k, start=1)
        if movie_id in relevant
    )
    ideal_dcg = math.fsum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1)
    )
    return dcg / ideal_dcg


def hit_rate_at_k(
    ranked_movie_ids: Iterable[int], relevant_movie_ids: Iterable[int], k: int
) -> float:
    """Return 1.0 if any of the first k movies is relevant, otherwise 0.0."""
    top_k, relevant = _inputs(ranked_movie_ids, relevant_movie_ids, k)
    return float(any(movie_id in relevant for movie_id in top_k))


def mean_metrics(per_user_results: Iterable[Mapping[str, Real]]) -> dict[str, float]:
    """Macro-average score-only mappings, giving every supplied user equal weight.

    All rows must have the same nonempty string keys and finite scores in [0, 1].
    Keep IDs and other run metadata outside these mappings. Matching user sets
    and checking missing/duplicate users belong to the prediction validator.
    """
    rows = list(per_user_results)
    if not rows:
        raise ValueError("per_user_results must contain at least one user")
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("each per-user result must be a mapping of metric names to scores")
    keys = set(rows[0])
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("metric keys must be nonempty strings, with at least one metric")
    values = {key: [] for key in sorted(keys)}
    for index, row in enumerate(rows):
        if set(row) != keys:
            raise ValueError(f"per-user result {index} has different metric keys")
        for key in values:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"per-user result {index}, {key}: score must be numeric, not Boolean")
            if not 0 <= value <= 1:
                raise ValueError(f"per-user result {index}, {key}: score must be finite and between 0 and 1")
            values[key].append(float(value))
    return {key: math.fsum(scores) / len(rows) for key, scores in values.items()}
