# Data pipeline — Member 1

## Phase 1: frozen dataset

`prepare.py` downloads, builds, and verifies the `ml25m-global-v1` dataset. `build.py` implements the global temporal split and catalog, and `verify.py` checks output hashes and history boundaries. Run from the repository root with `python -m src.data.prepare`. See [Sections 4 and 9 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).

## Phase 2: P2 training samples, catalog seed, and checks

These commands need the processed `ml25m-global-v1` folder **and** the extracted raw CSVs in `data/raw/ml-25m/` (titles, genres, and genome tags for out-of-catalog liked movies are rendered exactly as in the frozen profiles). They use the standard library only. Sample generation reads training ratings only; it never opens validation or test ratings, targets, or test profiles.

```bash
python -m src.data.p2_samples      # P2 samples + episodes, ~15 min
python -m src.data.catalog_seed    # catalog seed for Service A / vector index, ~1 min
python -m src.data.p2_checks       # data-quality and temporal-boundary report, exits 1 on any error
python -m src.data.p2_handoff      # ZIP of data + reports + field guide + SHA256SUMS (only if checks passed)
```

Outputs go to `data/processed/p2-v2/` (Git-ignored); the ZIP goes to `data/processed/handoff/`. Settings are in [`configs/p2_samples_v2.json`](../../configs/p2_samples_v2.json). Use `--train-users N --dev-users M` for a quick smoke run.

| Module | Purpose |
|---|---|
| `p2_samples.py` | Simulates per-user episodes (canonical, 25%/50% masked, or sparse profiles), labels actions by comparing action sequences on NDCG@10 (midterm §3.4), and emits one SFT sample per agent decision across the six action types, plus P1-format per-movie scoring samples. |
| `clarification_sim.py` | Frozen deterministic answer simulator `pre_cutoff_history_sim_v1`; answers only from ratings before the anchor/cutoff. |
| `p2_actions.py` | Reference action grammar, strict JSON parser, and `p2-trace-v1` episode-order validator. |
| `catalog_seed.py` | One record per catalog movie with canonical text, display fields, and training-only rating aggregates. |
| `p2_checks.py` | Re-derives every episode from boundary-truncated source ratings and checks leakage, masks, labels, grammar, scoring samples, and grounding. |
| `p2_handoff.py` | Packages the checked files with reports, field guide, and checksums for teammates. |
| `../profiles/agent_text.py` | Prompt renderers shared by training and inference: actions (`p2-prompt-v1`) and per-movie scoring (`p2-score-prompt-v1`, the P1 prompt plus clarifications). |

Method, label rules, results, limitations, and per-member handoff notes are in [docs/member1_p2_report.md](../../docs/member1_p2_report.md). Field definitions are in the [data dictionary](../../DATA_DICTIONARY.md#phase-2-artifacts--p2-v2) and the [handoff guide](../../docs/member1_p2_handoff.md).
