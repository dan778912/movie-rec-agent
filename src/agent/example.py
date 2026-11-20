"""Runnable demonstration for Member 3: how Service B can call the controller.

    python -m src.agent.example

EVERYTHING HERE IS A STAND-IN. ``DemoModel`` is a few rules reading the prompt, not a language
model, and ``DemoRetriever`` walks a tiny made-up catalog, not a vector index. They exist so the
calling pattern runs end to end with no weights, GPU or download. They show how the pieces connect,
not that any real model decides well. Replace them with the real checkpoint wrapper (Member 3/6)
and the real retriever (Member 3); the controller code does not change.

The example shows four things:
  1. a first call that ends in a clarifying question ("ask");
  2. the stateless way to continue: Service A replays the question and answer, Service B rebuilds
     the episode, retrieval widens the pool once, and the call ends in "recommend";
  3. the explicit-state way to continue: the state from call 1 is stored as JSON and resumed;
     it gives the same recommendations as (2);
  4. a degraded call: the model times out once, so the contract fallback (the retriever's own
     order) is returned, and the reason is recorded.
"""

from __future__ import annotations

import json
import re

from src.agent.config import AgentConfig
from src.agent.decide import Decision, RetrievalResult, decide, resume
from src.agent.trace import telemetry_record
from src.profiles.agent_text import serialize_action
from src.profiles.text import user_profile

# (title, genres, tags): ids are 1..N in this order. Made up for the demonstration.
MOVIES = [
    ("Night Heist (2003)", ["Crime", "Thriller"], ["heist", "twist ending"]),
    ("Paper Moon Road (1998)", ["Comedy", "Romance"], ["road trip", "quirky"]),
    ("The Long Cipher (2009)", ["Thriller", "Mystery"], ["codes", "slow burn"]),
    ("Harbor Lights (1994)", ["Drama", "Romance"], ["seaside", "melancholy"]),
    ("Iron Meridian (2011)", ["Action", "Thriller"], ["chase", "gadgets"]),
    ("Sunday Orchestra (2001)", ["Drama", "Musical"], ["ensemble", "uplifting"]),
    ("Glass Marathon (2016)", ["Crime", "Drama"], ["courtroom", "slow burn"]),
    ("Laughing Gas (1989)", ["Comedy"], ["slapstick", "dated"]),
    ("Cold Signal (2007)", ["Sci-Fi", "Thriller"], ["isolation", "twist ending"]),
    ("Velvet Alibi (1996)", ["Crime", "Mystery"], ["noir", "femme fatale"]),
    ("Moon Garden (2013)", ["Fantasy", "Drama"], ["lyrical", "slow"]),
    ("Quick Fuse (2005)", ["Action", "Comedy"], ["buddy cop", "explosions"]),
    ("Paper Trail (2012)", ["Thriller", "Drama"], ["whistleblower", "tense"]),
    ("Salt Road (1992)", ["Western", "Drama"], ["desert", "revenge"]),
    ("Blue Lantern (2018)", ["Sci-Fi", "Mystery"], ["space", "puzzle"]),
    ("Second Verse (2000)", ["Musical", "Comedy"], ["sing-along", "feel-good"]),
    ("The Quiet Wire (2014)", ["Thriller", "Crime"], ["surveillance", "slow burn"]),
    ("Dust Bowl Diaries (1997)", ["Drama", "War"], ["period", "bleak"]),
]
CATALOG = {i: {"movie_id": i, "title": t, "genres": g, "tags": tags} for i, (t, g, tags) in enumerate(MOVIES, start=1)}
LIKED = [{"movie_id": 1, "title": CATALOG[1]["title"], "rating": 5.0, "genres": CATALOG[1]["genres"], "tags": CATALOG[1]["tags"]},
         {"movie_id": 3, "title": CATALOG[3]["title"], "rating": 4.0, "genres": CATALOG[3]["genres"], "tags": CATALOG[3]["tags"]}]
HISTORY = [1, 3]  # the user has already rated these, so they are never recommended
PROFILE = user_profile(LIKED, [])  # the project's shared formatter, not hand-typed text

_LINE = re.compile(r"^\[(\d+)\] (.*?) \| genres: (.*?) \| tags: (.*)$")


def _action(task: str, **fields) -> str:
    return serialize_action({"action": task, **fields})


class DemoModel:
    """A stand-in for the checkpoint: reads the prompt, returns one JSON action."""

    def generate(self, prompt: str) -> str:
        task = re.search(r'Output format: \{"action": "(\w+)"', prompt).group(1)
        shown = [m for m in (_LINE.match(line) for line in prompt.splitlines()) if m]
        genres = {m[1]: set(m[3].split(", ")) for m in shown}
        liked_genres = {g for item in LIKED for g in item["genres"]}
        if task == "assess_sufficiency":
            return _action(task, sufficient="None so far." not in prompt)  # ask once, then proceed
        if task == "ask_clarification":
            return _action(task, question="Which movie genres do you enjoy most?")
        if task == "evaluate_candidates":
            return _action(task, pool_sufficient=len(shown) >= 14)  # a small pool is "not enough"
        if task == "retrieve_candidates":
            return _action(task, query="movies similar to " + ", ".join(item["title"] for item in LIKED))
        if task == "rank_candidates":
            order = sorted(genres, key=lambda movie_id: (-len(genres[movie_id] & liked_genres), int(movie_id)))
            return _action(task, movie_ids=[int(movie_id) for movie_id in order])
        explanations = []
        for movie_id, own in genres.items():
            source = next((item for item in LIKED if own & set(item["genres"])), None)
            if source:
                shared = sorted(own & set(source["genres"]))
                text = f"Like {source['title']}, which you rated {source['rating']:g}/5, it is listed under {' and '.join(shared)}."
            else:
                text = f"It is listed under {' and '.join(sorted(own))}."
            explanations.append({"movie_id": int(movie_id), "text": text})
        return _action(task, explanations=explanations)


class DemoRetriever:
    """A stand-in for Member 3's retrieval. It ignores the clarification answers (decision A7)."""

    index_version = "demo-index-v0"

    def initial_pool(self, *, profile_text, clarifications, history_movie_ids) -> RetrievalResult:
        ids = [i for i in CATALOG if i not in set(history_movie_ids)][:12]
        return RetrievalResult(ids, self.index_version)

    def search(self, *, query, exclude_movie_ids, max_new) -> RetrievalResult:
        ids = [i for i in CATALOG if i not in set(exclude_movie_ids)][: min(max_new, 6)]
        return RetrievalResult(ids, self.index_version)


class FlakyModel(DemoModel):
    """Raises a timeout on its first call, to show a degraded but explicit outcome."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("model server did not answer in time")
        return super().generate(prompt)


def main(*, show: bool = True) -> dict[str, Decision]:
    config = AgentConfig(label="example-not-a-real-run")
    request = {"profile_text": PROFILE, "history_movie_ids": HISTORY, "clarifications": []}
    call = dict(model=DemoModel(), retriever=DemoRetriever(), catalog=CATALOG)

    first = decide(request, config=config, **call)
    answer = "I mostly enjoy Thriller movies."

    # (2) Stateless: Service A resends what it stored, Service B keeps nothing between calls.
    replay = {**request, "clarifications": [{"question": first.response["question"], "answer": answer}]}
    replayed = decide(replay, config=config, **call)

    # (3) Explicit state: persist the state from call 1 as JSON and resume it.
    stored = json.dumps(first.state.to_dict())
    resumed = resume(json.loads(stored), answer, "demo_user", **call)

    # (4) A model timeout is an ordinary, recorded outcome that ends in the contract fallback.
    degraded = decide(request, config=config, model=FlakyModel(), retriever=DemoRetriever(), catalog=CATALOG)

    results = {"first": first, "replayed": replayed, "resumed": resumed, "degraded": degraded}
    if show:
        _print(results)
    return results


def _print(results: dict[str, Decision]) -> None:
    print("STAND-INS ONLY: DemoModel is a few rules, DemoRetriever walks a made-up catalog.\n")
    labels = {"first": "1. First call (no clarifications yet)",
              "replayed": "2. Second call, stateless: Service A replays the question and answer",
              "resumed": "3. Second call, explicit state: resume the JSON state from call 1",
              "degraded": "4. First call again, but the model times out once"}
    for key, decision in results.items():
        print(labels[key])
        response = decision.response
        if response["action"] == "ask":
            print(f"   ask: {response['question']!r}")
        elif response["action"] == "recommend":
            for item in response["items"][:3]:
                print(f"   #{item['rank']} movie {item['movie_id']} score {item['score']}: {item['explanation']}")
            print(f"   ... {len(response['items'])} items in all")
        else:
            print(f"   error: {response}")
        info = telemetry_record(decision.state)
        statuses = sorted({r["status"] for g in info["grounding"] for r in g["results"]})
        print(f"   ended: {info['termination_reason']}"
              + (f" ({info['reason']})" if info["reason"] else "")
              + f"; control events {info['control_events']}, retrievals {info['retrievals_used']}"
              + (f"; grounding screen: {', '.join(statuses)}" if statuses else "") + "\n")


if __name__ == "__main__":
    main()
