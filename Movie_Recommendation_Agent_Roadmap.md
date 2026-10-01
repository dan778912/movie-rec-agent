# Movie Recommendation Application — Team Project Plan

## 1. Executive Summary

This is an **application-focused NLP project**. The team will build and deploy a movie recommendation application that turns a user's rating history into text, retrieves promising movies, and uses a fine-tuned NLP model to rerank them.

The application will:

1. Accept movies and ratings from a user.
2. Convert that history into a natural-language preference profile.
3. Retrieve a candidate set from the MovieLens catalog.
4. Use a fine-tuned NLP model to score and rerank the candidates.
5. Return personalized recommendations with grounded explanations.
6. Run through a simple deployed web interface.

The goal is not to invent a new recommendation algorithm. The project demonstrates the course skills of constructing NLP inputs, fine-tuning a pretrained model, evaluating it against baselines, and deploying it in an application.

The team will build working baselines before investing in the application interface. This is a development order, not a reduction in the final application scope: **fine-tuning and deployment are required final deliverables**.

### How this satisfies the course requirement

| Course expectation | What this project will do |
|---|---|
| Application-focused | Build a usable personalized movie recommendation application |
| Use NLP concepts | Represent users and movies as text; use lexical retrieval, embeddings, and a text reranker |
| Train or fine-tune a model | Fine-tune a pretrained NLP model on MovieLens-derived relevance examples |
| Deploy models | Serve retrieval and the fine-tuned reranker through a working web application |
| Novel research not required | Use baselines to validate engineering choices rather than claim a new algorithm |

---

## 2. Scope

### Required core scope

- One dataset: MovieLens 25M
- Offline recommendation evaluation
- Popularity baseline
- BM25 keyword retrieval
- Embedding retrieval
- **A fine-tuned one-shot LLM reranker (P1)** — fine-tuned (LoRA on Qwen2.5-1.5B/3B) to rank retrieved candidates and generate justifications, no agentic behavior; tests whether fine-tuned LLM reranking helps over embedding order alone
- **A fine-tuned agentic NLP model (P2)** — the P1 model further fine-tuned to autonomously judge (1) whether the current preference information is sufficient, generating a clarifying question and looping if not, and (2) whether the retrieved candidate pool is good enough, triggering a wider re-retrieval if not; tests whether agentic multi-step reasoning helps over P1 — this agentic decision-making is the core proposed contribution, not an optional extension
- Recommendation explanation generation, treated as a core deliverable
- Reproducible Recall@K and NDCG@K comparisons
- A conversational multi-turn interaction interface (React + TypeScript)
- Two FastAPI microservices (API/business service + model inference service)
- A vector database (Chroma/Pinecone) for candidate retrieval
- PostgreSQL for conversation history, watch history, and recommendation records
- Docker + Docker Compose, GitHub Actions CI/CD, Terraform + AWS deployment

### Notes on scope

Microservices, PostgreSQL, a managed vector database, and Terraform/AWS deployment are part of the required architecture for this project (see the team's Notion project brief), not optional additions. React is required for the frontend, not merely allowed.

### Key terms

- **Baseline:** A simpler method used to judge whether the proposed system adds value.
- **Retrieval:** Quickly selecting a small candidate pool from the full movie catalog.
- **Reranking:** Reordering those candidates using a more detailed model.
- **Fine-tuning:** Updating a pretrained model's parameters using project-specific training examples.
- **Temporal split:** Training on earlier interactions and testing predictions against later interactions.
- **Deployment:** Making the complete pipeline usable through a running application rather than only a notebook.

### What a completed project looks like

By the deadline, the team should be able to show:

1. One reproducible MovieLens data split (Global Temporal Split).
2. Four working baselines in a shared results table (Popularity, BM25, Embedding retrieval, and P1).
3. B3 (embedding order) compared against P1 (one-shot fine-tuned reranker), isolating whether fine-tuned LLM reranking helps.
4. P1 compared against P2 (agentic fine-tuning: clarifying-question judgment, re-retrieval judgment, and candidate ranking + justification), isolating whether agentic multi-step reasoning helps further.
5. Saved and reloadable P1 and P2 model artifacts.
6. A deployed application (React + TypeScript frontend, two FastAPI microservices, vector database, PostgreSQL, Docker, Terraform + AWS) using the P2 model.
7. Documentation that lets another team member reproduce the main results and redeploy the app.

Agentic clarification and re-retrieval are part of the required core system, not bonus work — see Section 6, P2.

---

## 3. Dataset Decision

### Selected dataset: MovieLens 25M

Use only the files included in MovieLens 25M:

- `ratings.csv`: user, movie, rating, and timestamp
- `movies.csv`: title and genres
- `tags.csv`: user-generated movie tags
- `genome-tags.csv` and `genome-scores.csv`: tag vocabulary and movie-tag relevance
- `links.csv`: identifiers, if needed for inspection; external metadata is not required

### Why one dataset is viable

MovieLens 25M is sufficient to test the main retrieval and ranking claim. Ratings provide user histories and future relevant items. Titles, genres, tags, and tag-genome scores provide text-like metadata for BM25 and embedding retrieval.

Each movie will be represented using MovieLens information only. Example:

> Toy Story (1995). Genres: Adventure, Animation, Children, Comedy, Fantasy. Tags: Pixar, toys, friendship, family, imaginative.

Each user profile will be constructed only from the user's available historical ratings. Example:

> The user has strongly liked animated family adventures, Pixar movies, imaginative fantasy, and light comedy. Highly rated movies include Toy Story and Monsters, Inc.

### Limitation

MovieLens does not contain real recommendation conversations or human answers to clarifying questions. Therefore:

- Core results can support claims about retrieval and ranking quality.
- Clarification can only be evaluated through a controlled simulation.
- The report must not claim that simulated clarification proves improved real-user satisfaction.
- ReDial will not be merged into the core dataset because entity matching, domain differences, and incompatible supervision add risk before the central method is validated.

---

## 4. Shared Data and Evaluation Protocol

These rules must be frozen before model comparisons begin. Every method uses the same processed data, users, catalog, relevance definition, and evaluation code.

### 4.1 Eligible users and items

- Keep users with at least 20 ratings.
- Treat a rating of 4.0 or higher as a positive/relevant interaction.
- Never recommend a movie already present in the user's available history.
- Define the catalog from movies available before evaluation and with enough training activity to be scored reliably. The data owner will choose and document a threshold such as five training ratings.
- Record how many held-out positives fall outside the eligible catalog. This catalog-coverage loss must not be hidden.

### 4.2 Global Temporal Split

The original draft of this plan used a per-user chronological split (each user's own ratings split 80/10/10 by their own timeline). This is changed to a **Global Temporal Split**, because a per-user split has been shown to cause temporal leakage — each user gets a different train/test cutoff, so the model ends up training on interactions that occur, on the global timeline, later than interactions it is tested on. A global split closes this gap and better reflects the real task of predicting the future from the past:

- Zhao et al., "A Critical Study on Data Leakage in Recommender System Offline Evaluation," ACM TOIS: https://dl.acm.org/doi/full/10.1145/3569930
- Meng et al., "Recommender Systems: A Primer," Section on temporal splitting: https://arxiv.org/pdf/2302.02579
- "Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders," ACM RecSys 2025: https://arxiv.org/abs/2507.16289

Procedure:

1. Sort every rating in the entire dataset (across all users) by timestamp, oldest to newest.
2. Choose one global cutoff timestamp at the 90th-percentile point of that timeline.
3. All ratings before the cutoff go into training/validation; all ratings after the cutoff go into the test set.
4. Within the pre-cutoff portion, apply the same 90th-percentile logic again (or a simpler ratio) to separate training from validation.
5. Ensure every eligible user has at least one interaction on each side of the relevant cutoff after rounding, and document the exact rule in code.

Use the split in two stages:

- **Development/validation:** use training as user history and predict positive validation items.
- **Final testing:** after settings are frozen, use training plus validation as history and predict positive test items.

The test set must not be examined while choosing models, prompts, thresholds, pool sizes, or hyperparameters.

### 4.3 Preventing leakage

- Popularity counts, collaborative matrices, BM25 statistics, and learned representations must be fitted without test interactions.
- Test ratings cannot be included in user profiles.
- If tag-genome data is treated as static catalog metadata, apply that decision consistently to every text-based method.
- No method may receive ratings or history denied to another method unless that difference is the explicit subject of an ablation.

### 4.4 Evaluation population

Run inexpensive baselines on all eligible users when practical. Because neural reranking may be expensive, create one fixed evaluation subset using a recorded random seed. Every method in the reranker comparison must use that same subset.

Recommended starting point:

- 300 users for reranker development on validation data
- 500–1,000 fixed users for the final test comparison

Counts may change based on compute limits, but they must be selected before viewing final test results.

---

## 5. Shared Movie and User Representations

### Canonical movie document

Create one text document per movie containing:

- Title and year
- Genres
- A fixed number of highest-relevance genome tags
- Aggregated user tags from training data after cleaning

BM25, embedding retrieval, and the reranker should use this same document unless an experiment explicitly compares representations.

### Canonical user profile

Construct the profile from available history only:

- Include a fixed number of the highest-rated or most recent liked movies.
- Include genres and tags associated with liked movies.
- Optionally mention disliked movies separately, using the same rule for every user.
- Use deterministic formatting.

The first version should be template-generated, not LLM-generated. This prevents profile-generation variability from obscuring retrieval and ranking effects.

---

## 6. Systems to Compare

This project strictly follows the four comparison systems defined in the team's Notion project brief plus the proposed system (P2). No additional baseline (e.g., collaborative filtering) is included, to keep the comparison set identical to the proposal.

### B0 — Most Popular Movies

Rank unseen movies by the number of positive training ratings. Break ties deterministically.

Purpose: minimum sanity-check baseline — if a system can't beat this, something is broken.

### B2 — Keyword Retrieval (BM25)

Index canonical movie documents. Query them with the canonical user profile and return the highest-scoring unseen movies. No semantic understanding — only catches literal word overlap.

Purpose: measure semantic retrieval against literal term overlap.

### B3 — Embedding Retrieval

Encode movie documents and user profiles with a pretrained embedding model. Rank unseen movies by cosine similarity. No LLM involved.

Purpose: primary retrieval baseline and candidate generator for the proposed system; this is what the proposed method is compared against most directly.

### P1 — Fine-tuned LLM Reranking, One-Shot (No Agent)

Fine-tune Qwen2.5-1.5B/3B with parameter-efficient LoRA (SFT) to rank the top-50 candidates retrieved by B3 and generate a grounded justification for each recommendation. The agentic loop is disabled: the model never asks a clarifying question and never triggers re-retrieval — it always uses exactly the history and candidate pool it is given, in a single pass. Training data covers candidate-pool ranking + justification generation only.

**This isolates whether fine-tuned LLM reranking itself helps.** The primary comparison here is **B3 versus P1**: does reordering B3's candidates with a fine-tuned LLM beat B3's raw embedding-similarity order? If the available hardware cannot complete the documented smoke test, the predeclared fallback is a compact cross-encoder such as MiniLM. Prompt engineering alone (no parameter update) does not satisfy this deliverable.

### P2 — Agentic Multi-Step Reasoning (required proposed system)

Take the same P1 model and extend its fine-tuning with two additional training-example types: **information-sufficiency judgment** (should the agent ask a clarifying question given the current profile, or is it enough to proceed?) and **clarifying-question generation** (what should it ask?). At inference time, the agentic loop is enabled: the model autonomously decides — not via hard-coded rules — (1) whether the current preference information is sufficient, generating a clarifying question and looping if not, and (2) whether the retrieved candidate pool is good enough, triggering a wider re-retrieval if not, before producing the final ranking and justifications.

**This isolates whether agentic multi-step reasoning helps beyond one-shot LLM reranking.** The primary comparison here is **P1 versus P2**: does letting the model autonomously ask clarifying questions and re-retrieve improve results over the one-shot version, holding the base fine-tuned model constant? B3 versus P1 already established whether LLM reranking itself helps; P1 versus P2 isolates the additional, separate question of whether the agent's own decision-making adds further value.

#### Training-example construction (shared by P1 and P2's ranking sub-task)

For each eligible user:

1. Construct a profile using only ratings that occurred before the target rating.
2. Pair the profile with a candidate movie document.
3. Label ratings of 4.0 or higher as **relevant**.
4. Label explicit ratings of 2.5 or lower as **not relevant**.
5. Ignore ambiguous 3.0 and 3.5 ratings in the first version.
6. If unseen movies are sampled as weak negatives, mark and document them separately because "unrated" does not always mean "disliked."

Example:

```text
User profile:
Likes science fiction, space exploration, philosophical stories,
Arrival, and Interstellar.

Candidate movie:
2001: A Space Odyssey.
Genres: Adventure, Drama, Science Fiction.
Tags: space travel, artificial intelligence, philosophical.

Training target:
Relevant
```

P2's additional information-sufficiency and clarifying-question training examples are constructed by holding out part of a user's profile (simulating incomplete information) and using the held-out preferences to write the target clarifying question and its answer.

Training examples must come from the training portion only. Validation examples select the checkpoint, hyperparameters, and decision threshold. Test interactions are used once for final reporting.

At inference time, both P1 and P2 rerank the same candidate pools produced by B3. This isolates the effect of each added capability. The pool size may be tuned on validation data but must be frozen before testing. Explanations are generated by a constrained decode that only sees known user history and MovieLens metadata. Because MovieLens has no human-written explanation labels, explanation generation is evaluated for grounding in addition to being a required deliverable.

### Deeper ablations within P2 (secondary, time-permitting)

Once P1 versus P2 is established, optionally decompose P2's contribution further:

- **Clarification-only ablation:** P2 with clarification enabled but re-retrieval disabled, versus P1.
- **Re-retrieval-only ablation:** P2 with re-retrieval enabled but clarification disabled, versus P1.

Only one component changes in each ablation relative to P1. Do not combine clarification, re-retrieval, and a new embedding model in one comparison.

---

## 7. Metrics and Reporting

### Candidate retrieval

- Recall@20
- Recall@50
- Recall@100 if 100 candidates are returned

These measure whether relevant items enter the candidate pool. A reranker cannot recover a movie that retrieval never supplies.

### Final ranking

- Recall@10
- NDCG@10
- Optional Hit Rate@10 as a secondary, easy-to-explain metric

Use binary relevance in the core experiment: held-out ratings of 4.0 or higher are relevant. A graded-relevance experiment may be added but must be reported separately.

### Required reporting

- Mean metric across users
- Number of evaluated users
- Catalog coverage and excluded targets
- Candidate-pool size
- Model names and versions
- Input-template or prompt version and inference settings
- Training time, inference runtime, and approximate compute/API cost
- Paired bootstrap confidence intervals for important differences, especially P1 versus P2

Do not treat a tiny numerical increase as meaningful without checking variability across users.

---

## 8. Shared Interfaces and Repository Contract

### Repository structure (already scaffolded)

The repository has already been scaffolded (with `.gitignore`, `requirements.txt`, and a per-folder `README.md` stub pointing back to the plan section it implements) and is ready to `git init` and push:

```text
movie-rec-agent/
├── README.md                    Setup and usage instructions
├── requirements.txt              Python dependencies (retrieval, LoRA fine-tuning, FastAPI, etc.)
├── .gitignore                    Excludes raw data, models, and prediction outputs from git
├── configs/                      Frozen experiment configurations
├── data/
│   ├── raw/                      Unmodified MovieLens 25M files (not committed — see .gitignore)
│   └── processed/                Versioned split and metadata artifacts
├── src/
│   ├── data/                     Preprocessing + Global Temporal Split (Section 4.2)
│   ├── profiles/                 Canonical movie-document and user-profile construction (Section 5)
│   ├── baselines/                B0 popularity baseline (Section 6)
│   ├── retrieval/                B2 BM25 and B3 embedding retrieval (Section 6)
│   ├── reranking/                P1 one-shot reranker + P2 agentic fine-tuning (Section 6)
│   ├── agent/                    P2's clarification + re-retrieval loop (Section 6)
│   ├── app/                      React + TypeScript frontend, FastAPI business + inference services
│   └── evaluation/                Recall@K, NDCG@K, paired bootstrap CIs, result tables (Section 7)
├── models/                       Fine-tuned adapter/checkpoint references (P1, P2)
├── outputs/
│   ├── predictions/               Standard JSONL prediction files, one per system
│   └── reports/                   Generated result tables and plots
└── tests/                         Unit and smoke tests
```

Each member owns the folder(s) matching their Section 9 workload (e.g., Member 4 owns `src/retrieval/`, Member 6 owns `src/reranking/`), but reviews are cross-cutting per Section 9's note that every artifact has one accountable owner while teammates review each other's work.

### Standard prediction record

Every system outputs JSONL with one record per user:

```json
{
  "user_id": 123,
  "split": "test",
  "method": "embedding",
  "recommendations": [
    {"movie_id": 50, "score": 0.92, "rank": 1},
    {"movie_id": 260, "score": 0.87, "rank": 2}
  ]
}
```

LLM methods may add an `explanation` field, but metrics depend on IDs and ranks, not free-form text.

### Reproducibility requirements

Every experiment saves:

- Configuration file
- Git commit or code version
- Random seed
- Input split version
- Output predictions
- Summary metrics
- Error/failure count

No member should calculate final metrics in an isolated notebook. The shared evaluator is authoritative.

---

## 9. Six Current Modeling Workloads

These are the six workloads for the current baseline-first phase. They are intentionally not identical in size. Owners should review one another's work, but every artifact has one accountable owner. No member is assigned application or deployment work during this phase. After the modeling gate is passed, the team will regroup and divide application work separately.

### Member 1 — Data and representation owner

**Goal:** Produce frozen data artifacts consumed by everyone else.

**Tasks:**

- Download and document MovieLens 25M.
- Filter eligible users and catalog items.
- Implement the Global Temporal Split (Section 4.2).
- Construct canonical movie documents.
- Construct deterministic validation and test user profiles.
- Add leakage checks and dataset statistics.
- Assign a version to the processed dataset.

**Deliverables:**

- Re-runnable preprocessing command
- Split files, catalog, movie documents, and user profiles
- Data dictionary
- User, item, interaction, positive, sparsity, and coverage statistics
- Tests confirming test interactions do not appear in profiles

**Done when:** A teammate can run one documented command and reproduce the same counts and checksums.

### Member 2 — Evaluation and experiment owner

**Goal:** Make every method comparable under one trusted evaluator.

**Tasks:**

- Implement Recall@K, NDCG@K, and optional Hit Rate@K.
- Validate metrics with hand-calculated perfect and bad rankings.
- Enforce watched-item removal.
- Create the fixed reranker evaluation subset and seed.
- Implement paired bootstrap confidence intervals.
- Track invalid IDs, duplicates, missing users, and coverage.
- Generate the final comparison table from standard predictions.

**Deliverables:**

- Evaluation command and unit tests
- Prediction-file validator
- Automatically generated results table
- Written experiment protocol

**Done when:** The evaluator accepts every compliant method and rejects malformed or unfair outputs with clear errors.

### Member 3 — Popularity baseline, and Member 6's designated supporter

**Goal:** Support Member 6 first, since that work blocks the project's highest-risk task, then establish the popularity sanity-check baseline.

**Tasks:**

- **As soon as Member 1's processed data is available**, support Member 6 by: converting it into profile–candidate training examples for P1's ranking task; checking label counts and class balance on those training examples; and auditing a sample of P1's generated explanations for unsupported claims. This comes first because Member 6's fine-tuning cannot start without it, and it is the project's longest, highest-risk critical path.
- Implement popularity ranking (B0). This comes after the support work, since B0 is quick regardless of when it's done and does not block anyone else.
- Score unseen catalog items for fixed evaluation users.
- Document compute time and limitations.

**Deliverables:**

- P1 training-example file, with label-balance report
- Explanation-faithfulness audit
- B0 prediction file
- Reproducible scoring code
- Validation sweep summary
- Short failure analysis

**Done when:** Both baselines pass the shared validator and appear in the results table.

### Member 4 — BM25 retrieval owner

**Goal:** Establish the keyword-retrieval baseline using the shared text representations.

**Tasks:**

- Implement BM25 over canonical movie documents.
- Remove watched movies.
- Tune BM25 parameters and document fields on validation data.
- Measure candidate recall and retrieval latency.
- Analyze cases where literal keyword matching succeeds or fails.

**Deliverables:**

- B2 prediction files
- Reproducible BM25 indexing and retrieval commands
- Frozen BM25 configuration
- Candidate-recall and latency report
- Short BM25 failure analysis

**Done when:** BM25 produces valid recommendations for every evaluation user and appears in the shared results table.

### Member 5 — Embedding retrieval owner

**Goal:** Build the semantic retriever and provide the fixed candidate pools used by the proposed reranker.

**Tasks:**

- Select and document a pretrained embedding model.
- Encode canonical movie documents and user profiles.
- Cache embeddings and build a local nearest-neighbor index.
- Remove already-watched movies.
- Tune the embedding choice and candidate-pool size on validation data.
- Measure Recall@20/50/100 and retrieval latency.
- Freeze and export identical top-50 candidate pools for Member 6.
- Compare semantic retrieval directly with Member 4's BM25 results.

**Deliverables:**

- B3 prediction files
- Reproducible embedding and indexing commands
- Model and configuration identifiers
- Cached embedding/index artifacts
- Frozen top-50 candidate files
- BM25-versus-embedding comparison

**Done when:** Every evaluation user has a valid frozen candidate pool that Member 6 can consume without custom conversion.

### Member 6 — Reranking and fine-tuning owner

**Goal:** Build and evaluate a fine-tuned reranking model that reorders a fixed set of embedding-retrieved movie candidates.

**Tasks:**

- Implement the one-shot reranker.
- Run a small fine-tuning smoke test before full training, using Member 3's training-example files.
- Fine-tune the reranker with LoRA on Qwen2.5-1.5B/3B (or the approved cross-encoder fallback).
- Select checkpoints and hyperparameters using validation data only.
- Score Member 5's exact candidate pools, producing standard prediction files.
- Generate grounded explanations alongside each ranking (auditing is Member 3's job, per Section 9).
- Record the model, configuration, seed, hardware, training time, latency, and cost.

**Deliverables:**

- Reproducible training command and configuration
- Saved adapter/checkpoint or documented model artifact
- Reranker prediction file
- Training and validation curves
- Failure and cost report

**Done when:** Another teammate can load the trained reranking model, reproduce its predictions, and use the shared evaluator to compare them with the original embedding ranking.

### Work after the six modeling workloads

After the team understands the baseline and fine-tuning results, it will divide application integration and deployment work in a new phase. Optional clarification and re-retrieval experiments must not delay that required final application work.

---

## 10. Dependency and Collaboration Plan

The team's time is organized into two explicit phases, each with a clear gate before moving to the next. This keeps the whole team engaged in each phase rather than leaving Member 6 to work through P1 and P2 alone while everyone else is idle.

### Phase 1 — Validate whether LLM reranking helps (B3 vs. P1)

**Estimated duration: 1–2 weeks.**

**Goal:** by the end of this phase, the team has a working, evaluated answer to "does fine-tuned LLM reranking beat raw embedding order?"

**Stage A — Freeze the contract**
Members 1 and 2 finalize the split, catalog, relevance labels, profiles, prediction format, and metric tests. Members 3–6 review the contract before it is frozen.

**Stage B — Build baselines and P1 in parallel**
- Member 3: as soon as Member 1's data is available, support Member 6 with training-data construction, output validation, and the explanation-faithfulness audit — this comes first, since it blocks Member 6's fine-tuning and sits on the phase's longest critical path. Popularity (B0) comes after, since it's quick regardless of timing and doesn't block anyone.
- Member 4: BM25 retrieval
- Member 5: embedding retrieval and candidate-pool generation
- Member 6: training-example generator for P1's ranking task, one-shot P1 fine-tuning, and a fine-tuning smoke test
- Members 1, 4, and 5 are not pulled into supporting Member 6: Member 1's data pipeline is the highest-stakes deliverable in this phase (everyone else depends on it, and any bug found later means revisiting it), and Members 4 and 5's own baselines (BM25, embedding retrieval) are genuine ongoing deliverables, not idle time.

**Stage C — Produce the first comparison**
- Members 3, 4, and 5 submit standard validation predictions.
- Member 2 generates the first shared table (B0–B3).
- The team selects the embedding model and pool size using validation results only.

**Stage D — Test P1**
- Member 5 freezes and exports candidate pools.
- Member 6 evaluates P1 on those exact pools.
- Member 2 compares embedding order (B3) with P1.
- The team performs error analysis before moving to Phase 2.

**Gate — Phase 1 complete when:** the B3-versus-P1 result is understood and reported (whether or not P1 wins), per Gate 3's criteria in Section 11.

### Phase 2 — Validate and implement the agent (P1 vs. P2)

**Workload assignment for this phase will be decided after Phase 1's results are in**, not fixed in advance — Phase 1's outcome (how well P1 performs, which team members are free, how much time remains) should inform how Phase 2's work is actually split.

**Goal:** this phase is primarily about two things — **verification** (does agentic multi-step reasoning, i.e. clarifying questions and re-retrieval, help beyond one-shot P1?) and **implementation** (build P2 and integrate it into a working, deployed application). Gate 3 and Gate 4 in Section 11 define what "done" looks like for each.

### Stage F — Integration, final evaluation, and demonstration

- The team integrates P2 into the full deployed pipeline.
- Member 2 runs the frozen final test comparison across B0–B3, P1, and P2.
- The team records application latency, known failures, and qualitative examples.
- The final demonstration uses the deployed P2 pipeline.

---

## 11. Baseline-First Milestones and Decision Gates

### Gate 1 — Data and evaluator ready

- Frozen split and catalog
- Leakage tests pass
- Metric unit tests pass
- Prediction validator works
- Shared experiment configuration exists

### Gate 2 — Working baseline table

Required before substantial frontend, database, microservice, or cloud work:

| Method | Recall@10 | Candidate Recall@50 | NDCG@10 | Users | Coverage |
|---|---:|---:|---:|---:|---:|
| Popularity | TBD | N/A | TBD | TBD | TBD |
| BM25 | TBD | TBD | TBD | TBD | TBD |
| Embedding retrieval | TBD | TBD | TBD | TBD | TBD |

The table must be generated by the shared evaluator from saved prediction files.

### Gate 3 — P1 and P2 ready

Compare:

- B3's top 10 from its top-50 pool
- P1's one-shot fine-tuned top 10 from that exact pool
- P2's agentic top 10, produced from the same B3 pool as its starting point (P2 may widen the pool via re-retrieval if it judges the pool insufficient)

The fine-tuning deliverable is complete when both the P1 and P2 model artifacts can be loaded reproducibly and their test predictions have been evaluated. Ideally P1 improves NDCG@10 over B3 (LLM reranking helps), and P2 improves NDCG@10 over P1 (agentic reasoning helps further), each with a meaningful paired confidence interval. If either does not hold, report and analyze that result; successful completion does not require inventing a positive result.

### Gate 4 — Application deployed

- A user can enter or select movie preferences, through multi-turn conversation if P2's clarification step triggers.
- The application builds a textual preference profile.
- Retrieval supplies candidates.
- P2 (the agentic fine-tuned model) reranks them, asking a clarifying question or re-retrieving when it judges that necessary.
- The interface returns recommendations and grounded explanations.
- An instructor or teammate can access the deployment.
- Setup, model-loading, and redeployment instructions have been tested.
- End-to-end latency and known limitations are documented.

### Gate 5 — Agent claims bounded by evidence

For clarification and re-retrieval, document the control (P1), the changed condition (P2, or a single-capability ablation), the metric, decision frequency, failure rate, and simulation limitations. This gate is required, since P2 is the required proposed system, not optional.

---

## 12. Minimum Final Results Table

| System | Personalized? | Text semantics? | Fine-tuned? | Agent loop? | Recall@10 | NDCG@10 |
|---|---|---|---|---|---:|---:|
| Popularity (B0) | No | No | No | No | TBD | TBD |
| BM25 (B2) | Yes | Keyword | No | No | TBD | TBD |
| Embedding retrieval (B3) | Yes | Semantic | No | No | TBD | TBD |
| One-shot fine-tuned reranker (P1) | Yes | Semantic | Yes | No | TBD | TBD |
| Agentic reranker (P2) | Yes | Semantic | Yes | Yes | TBD | TBD |

Also report candidate recall, confidence intervals, coverage, runtime, LLM cost, and qualitative failures.

---

## 13. Risks and Fallbacks

### Fine-tuning does not improve recommendation metrics

This is a valid application-development result as long as training was implemented correctly and reported honestly. Analyze the data, labels, profile quality, candidate metadata, class balance, and model capacity. The team still demonstrates model adaptation and deployment; it must not tune against the final test set to manufacture an improvement.

### Qwen fine-tuning exceeds available compute

Run an early smoke test on a small sample. If it cannot fit the available hardware or project timeline, use the predeclared MiniLM cross-encoder fallback for P1's ranking sub-task only. MiniLM is a classifier, not a generative model — it cannot produce clarifying questions or make the autonomous decisions P2 requires, so this fallback lets the team still report a valid B3-versus-P1 result, but P2 (the required agentic system) would need Qwen (or another generative model) at some scale to exist at all. Record the reason for the change.

### LLM evaluation is too expensive

Use the predetermined subset, a smaller open model, cached outputs, and one validation pass per prompt version. Never remove users after observing unfavorable results.

### Deployment cannot host the full model

Prefer a smaller model or adapter and quantization to fit hosting constraints, while keeping the required architecture (React + TypeScript frontend, two FastAPI microservices, vector database, PostgreSQL, Docker, Terraform + AWS) intact — do not substitute a lighter-weight interface or a different vector index as a way to skip the required stack. Provide a tested Docker and Terraform setup and confirm the acceptable deployment definition with the instructor. A notebook alone is not the planned deliverable.

### Clarification simulation feels artificial

The clarification and re-retrieval capabilities are required parts of P2, not a droppable side claim — this risk cannot be resolved by removing them from the central claim. If the simulated clarification signal looks too artificial to be convincing, refine the simulation method (e.g., how held-out preferences are converted into a target question) or narrow the claim to what the simulation can actually support, and report that limitation honestly. The retrieval-and-reranking (B3 vs. P1) result remains valid regardless.

### Explanations hallucinate unsupported details

Restrict explanations to supplied metadata, ask them to refer to known liked movies or tags, and report a manual faithfulness audit. Explanations are not evidence of recommendation accuracy.

---

## 14. Minimum Deployable Application

The required application flow is:

1. A user searches the MovieLens catalog and enters ratings for several movies.
2. The system builds the same deterministic profile used in evaluation.
3. Embedding retrieval returns candidates.
4. The fine-tuned reranker orders them.
5. The interface displays recommendations and grounded explanations.
6. The application records enough timing and error information to diagnose failures.

The application accepts any number of ratings (even just one or a few) — there is no hard-coded minimum. Whether the entered ratings are enough to make a good recommendation is P2's own judgment call, not an application-level rule: if P2 judges the profile insufficient, it asks a clarifying question before proceeding; if it judges the profile sufficient (even from a small number of clear ratings), it goes straight to retrieval and recommendation. Enforcing a fixed minimum in the UI would short-circuit exactly the decision P2 is meant to make. The application must exclude movies the user already rated. This live user does not need to be an existing MovieLens user; their entered ratings are converted into the same profile format used during training and evaluation.

### Required technical architecture

```text
React + TypeScript frontend
              |
   FastAPI Service A (API / business logic)
              |
       PostgreSQL (conversation, watch
       history, recommendation records)
              |
   FastAPI Service B (model inference)
              |
   Vector database (Chroma/Pinecone)
              |
   Fine-tuned agentic NLP model (P2)
              |
recommendations + grounded explanations
```

The business service and inference service are deployed as two independent FastAPI microservices (split because inference needs GPU and business logic doesn't, so they scale independently). Deployment uses Docker + Docker Compose locally, with Terraform provisioning AWS resources and GitHub Actions handling CI/CD for the production deployment.

### Minimum acceptance test

A teammate who did not build the interface must be able to:

1. Open the deployed application.
2. Select and rate a small number of catalog movies (try both a case with few, ambiguous ratings that should trigger a clarifying question, and a case with enough clear ratings that should go straight to recommendations).
3. Submit the ratings without editing code or files.
4. Receive 10 unseen recommendations.
5. See a short explanation grounded in the entered preferences and MovieLens metadata.
6. Confirm from an application/version label that the fine-tuned model produced the ranking.
7. Repeat the flow for a different preference profile without restarting the service.

### Required final artifacts

- Source code and dependency/environment specification
- Processed-data and model-building instructions
- Fine-tuning configuration and saved adapter/checkpoint reference
- Frozen evaluation results
- Runnable application
- Deployment URL or instructor-approved accessible deployment
- Setup and redeployment instructions
- Short demonstration script
- Limitations and responsible-claims section

If optional agent ablations are supported, the interface may ask a clarifying question or show that it broadened retrieval. The demo must first expose the required evaluated and fine-tuned pipeline.
