# Data pipeline — Member 1

## Phase 1: frozen dataset

`prepare.py` downloads, builds, and verifies the `ml25m-global-v1` dataset. `build.py` implements the global temporal split and catalog, and `verify.py` checks output hashes and history boundaries. Run from the repository root with `python -m src.data.prepare`. See [Sections 4 and 9 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).

## Phase 2: P2 training samples, catalog seed, and checks

These commands need the processed `ml25m-global-v1` folder **and** the extracted raw CSVs in `data/raw/ml-25m/` (titles, genres, and genome tags for out-of-catalog liked movies are rendered exactly as in the frozen profiles). They use the standard library only and never open `test_ratings.csv`, `test_targets.csv`, or `test_profiles.jsonl`.

```bash
python -m src.data.p2_samples      # P2 samples + episodes, ~2 min
python -m src.data.catalog_seed    # catalog seed for Service A / vector index, ~1 min
python -m src.data.p2_checks       # data-quality and temporal-boundary report, exits 1 on any error
```

Outputs go to `data/processed/p2-v1/` (Git-ignored). Settings are in [`configs/p2_samples_v1.json`](../../configs/p2_samples_v1.json). Use `--train-users N --validation-users M` for a quick smoke run.

| Module | Purpose |
|---|---|
| `p2_samples.py` | Simulates per-user episodes and emits one SFT sample per agent decision across the six action types. |
| `clarification_sim.py` | Frozen deterministic answer simulator `pre_cutoff_history_sim_v1`; answers only from ratings before the anchor/cutoff. |
| `p2_actions.py` | Reference action grammar, strict JSON parser, and `p2-trace-v1` episode-order validator. |
| `catalog_seed.py` | One record per catalog movie with canonical text, display fields, and training-only rating aggregates. |
| `p2_checks.py` | Re-derives every episode from boundary-truncated source ratings and checks leakage, labels, grammar, and grounding. |
| `../profiles/agent_text.py` | Prompt renderer shared by training and inference (`p2-prompt-v1`). |

Method, label rules, results, limitations, and per-member handoff notes are in [docs/member1_p2_report.md](../../docs/member1_p2_report.md). Field definitions are in the [data dictionary](../../DATA_DICTIONARY.md#phase-2-artifacts--p2-v1).
