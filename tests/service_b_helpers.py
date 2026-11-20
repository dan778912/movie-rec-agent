"""Explicit test doubles, not recommendation models."""
import json
import re

from src.agent.decide import RetrievalResult


class ScriptedModel:
    ready = True

    def __init__(self, ask=False, widen=False, invalid_once=False):
        self.ask, self.widen, self.invalid_once = ask, widen, invalid_once

    def generate(self, prompt):
        if self.invalid_once:
            self.invalid_once = False
            return "not JSON"
        task = re.search(r'Output format: \{"action": "(\w+)"', prompt)[1]
        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", prompt, re.M)]
        if task == "assess_sufficiency":
            fields = {"sufficient": not self.ask or "None so far." not in prompt}
        elif task == "ask_clarification":
            fields = {"question": "Which movie genres do you enjoy most?"}
        elif task == "evaluate_candidates":
            fields = {"pool_sufficient": not self.widen or len(ids) > 50}
        elif task == "retrieve_candidates":
            fields = {"query": "Comedy movies"}
        elif task == "rank_candidates":
            fields = {"movie_ids": list(reversed(ids))}
        else:
            fields = {"explanations": [{"movie_id": mid, "text": "It is listed under Comedy."} for mid in ids]}
        return json.dumps({"action": task, **fields})


def catalog(size=80):
    return {i: {"movie_id": i, "title": f"Film {i} (2000)", "genres": ["Comedy"], "tags": [],
                "text": f"Film {i} Comedy", "genome_tags": [], "user_tags": [], "decade": "2000s"}
            for i in range(1, size + 1)}


class ScriptedRetriever:
    index_version = "test-index-v1"

    def initial_pool(self, *, profile_text, clarifications, history_movie_ids):
        return self.search(query=profile_text, exclude_movie_ids=history_movie_ids, max_new=50)

    def search(self, *, query, exclude_movie_ids, max_new):
        return RetrievalResult([i for i in range(1, 81) if i not in exclude_movie_ids][:max_new], self.index_version)
