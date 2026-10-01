# Data pipeline — Member 1

`prepare.py` downloads, builds, and verifies the `ml25m-global-v1` dataset. `build.py` implements the global temporal split and catalog, and `verify.py` checks output hashes and history boundaries. Run from the repository root with `python -m src.data.prepare`. See [Sections 4 and 9 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).
