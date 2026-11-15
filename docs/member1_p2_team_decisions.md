# P2 training data — decisions for the team (Member 1)

**Status:** open for team decision. Members 2–6 should approve or change each item before P2 validation runs; it must be frozen before any P2 test inference. Background and full results: [member1_p2_report.md](member1_p2_report.md).

Each item lists what is implemented now (v1), why, and the alternative. Changing anything here means editing `configs/p2_samples_v1.json` (or the named module) and rerunning the three commands — about 4 minutes.

## D1. How "ask?" and "re-retrieve?" labels are decided — **conflicts with the midterm report**

Midterm §3.4 says a builder "will compare sampled action sequences using held-out interactions within the training period, choosing higher NDCG@10 and fewer actions on ties." v1 does something else:

- **v1 rule (implemented):** ask when the best simulated answer would raise the cosine similarity between the preference vector and the held-out positives by ≥ 0.04. Re-retrieve when a fresh retrieval would beat the current pool by ≥ 0.05 on the same measure.
- **Midterm rule, measured on 1,000 of the same training episodes:**

| | Midterm NDCG@10 rule | v1 cosine-gain rule |
|---|---:|---:|
| Starting states with NDCG@10 = 0 (all comparisons tie) | 80% | — |
| "Ask" rate, full (canonical) profile | 18% | 22% |
| "Ask" rate, sparse (1–8 rating) profile | 20% | 77% |
| Agreement between the two rules on "ask?" | 50% | |
| "Re-retrieve" rate at first pool check | 6.6% | 25% |

Held-out positives rarely land in any top 10, so the NDCG rule mostly records ties ("don't ask"). Its ask decisions barely depend on how much the profile shows, so the model would learn little. The cosine-gain rule gives a smooth signal and the intended behaviour: thin profiles ask much more.

**Recommendation:** keep v1 and update the final report's method text to describe it. It is still a decision about the value of an action, judged against held-out preferences inside the training period. **Alternative:** I can add the NDCG rule as a config option, keeping the measured imbalance in view.

## D2. Thresholds 0.04 / 0.05

These were chosen by inspecting label balance on a 150-user, training-period-only smoke run. No validation or test outcome was used. Raising them makes the agent ask and retrieve less. **Decide:** accept, or choose new values using validation P2 behaviour (allowed by the protocol). They must be frozen before test.

## D3. Loop budgets and pool sizes

- **Budgets:** at most 2 questions and 1 retrieval (8 control events). This matches the midterm and the `p2-trace-v1` cap. ✓
- **Pool sizes — differ from the plan:** training uses a 20-movie pool plus 20 re-retrieved (40 maximum). The midterm says 50 → at most 100, and offline evaluation starts from the frozen B3 50.
- **Prompt length:** rank prompts are about 9.4k characters (~2.5k tokens) now. At 50 + 50 they would be about 4–5k tokens.

**Decide with Member 6:** match evaluation (`pool_size: 50`, `retrieval_k: 50`) if training fits that length, or keep 20/20 and have Member 4's agent show the model the pool in chunks.

## D4. Clarification simulator `pre_cutoff_history_sim_v1`

- **Intents:** liked example, favorite genres, themes, dislikes, era.
- **Answers:** drawn from history before the cutoff only, preferring facts the profile text does not already show.
- **Fallbacks:** fixed answers for unrecognized questions or when nothing is left to reveal.

The protocol requires one frozen simulator. **Decide:** whether Member 4's offline P2 runner uses this module (recommended — training and evaluation then share one implementation), and freeze it before P2 validation scoring.

## D5. Prompt template `p2-prompt-v1` and action format

Training prompts come from `src/profiles/agent_text.py`. Inference must render them identically, and outputs must parse with `src/data/p2_actions.py`. **Decide (Members 4, 6):** adopt as is, or request changes now. Any later change needs new samples and a new version string.

## D6. Labels forced by the caps

- **What gets forced:** 218 training episodes still wanted a third question, and 318 of 1,006 retrievals still wanted a second. Those final decisions are labelled "proceed", matching what the loop cap would do.
- **How to find them:** they are flagged in `provenance.label_basis`.

**Decide (Member 6):** keep, down-weight, or drop.

## D7. Profile mismatch at test time — for Members 3, 6 and 2

| Profiles | With ≥ 1 out-of-catalog liked movie | Out-of-catalog share of liked entries |
|---|---:|---:|
| Frozen validation evaluation profiles | 174 / 4,751 (3.7%) | 0.5% |
| Frozen test evaluation profiles | 1,172 / 3,779 (31.0%) | 6.7% |
| P2 training samples, canonical | 2 / 1,959 (0.1%) | 0.01% |
| P2 validation samples, canonical | 16 / 307 (5.2%) | 0.7% |
| Member 3's P1 training samples | 0% by construction (catalog-only profiles) | 0% |

- **Cause:** 2,412 of the 2,520 out-of-catalog liked entries in test profiles were rated after the validation cutoff. They are newer movies that the training-defined catalog cannot contain.
- **What changes at test:** about a third of test users show the model titles that appear in no training document or candidate list.
- **Who is affected:** both P1 and P2 training data under-represent this, so it is a train/test shift, not a P2 format issue.
- **Options:**
  - (a) Accept it and report it as a limitation (recommended; no test data can be used to fix it).
  - (b) Run a validation-only diagnostic: re-score P1 with out-of-catalog likes removed from profiles, to see if the shift matters. Use validation users only, because P1/B3 test results are already known.
- **Not an option:** changing the frozen profile rule now would invalidate the published P1/B3 comparison.

## D8. Claims about simulated clarification

MovieLens has no real conversations. The answers and the ask/retrieve labels come from a deterministic simulation over each user's earlier ratings. The final report may claim effects under that **controlled offline simulation** only. It must not claim better real-user satisfaction or question quality. It should report how many questions were asked, how many answers revealed a new fact (74% in training), and that some labels were forced by the caps. This matches midterm §3.4 and the Phase 2 protocol.

## Decision record

| Item | Decision | Who / date |
|---|---|---|
| D1 label rule | Pending | |
| D2 thresholds | Pending | |
| D3 pool sizes | Pending | |
| D4 simulator | Pending | |
| D5 prompt/action format | Pending | |
| D6 forced labels | Pending | |
| D7 profile shift handling | Pending | |
| D8 claim wording | Pending | |
