"""Reproducible HTTP smoke against an explicitly labelled mock Service B."""
import argparse
import json
import time
from pathlib import Path

import httpx


def smoke(base_url, ready_timeout=60):
    with httpx.Client(base_url=base_url, timeout=5) as client:
        deadline = time.monotonic() + ready_timeout
        while True:
            try:
                response = client.get("/health/ready")
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.25)
        if response.json()["mode"] != "mock":
            raise ValueError("This smoke is for demonstration mode; it must not trigger real inference")
        request = {"profile_text": "I like thrillers", "history_movie_ids": [1, 3], "clarifications": []}
        first = client.post("/decide", json=request)
        first.raise_for_status()
        assert first.json()["action"] == "ask"
        answer = "I mostly enjoy Thriller movies."
        second = client.post("/decide", json={**request, "clarifications": [{"question": first.json()["question"], "answer": answer}]})
        second.raise_for_status()
        assert len(second.json()["items"]) == 10
        begin = client.post("/v1/decide", json=request)
        begin.raise_for_status()
        end = client.post("/v1/resume", json={"continuation": begin.json()["continuation"], "answer": answer})
        end.raise_for_status()
        assert begin.json()["trace_id"] == end.json()["trace_id"]
        assert end.json()["decision_trace"]["events"][-1]["action"] == "explain_recommendations"
        assert not {item["movie_id"] for item in end.json()["items"]} & {1, 3}
        return {"status": "passed", "mode": "mock", "legacy": "ask/recommend",
                "versioned": "signed continuation/full episode trace", "recommendation_count": 10}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--output")
    args = p.parse_args()
    report = smoke(args.url)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
