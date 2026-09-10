# Fraud Detection via Relational Machine Learning

A production-grade, scientifically disciplined machine learning and relational graph intelligence system for financial fraud detection, benchmarked on the IEEE-CIS Fraud Detection dataset (590,540 transactions).

---

## 1. Project Overview & Scientific Architecture

This repository explores whether **relational structure** (card identities, entity relationships, repeat transaction bursts, and graph neighborhood topology) can meaningfully enhance tabular gradient boosting models without introducing target leakage or temporal data snooping.

```
fraud-relational-ml/
├── 1. Data Ingestion & Strict Split Protocol (70/15/15 Temporal)
│     └── model_dataset.parquet (590,540 rows, 435 raw features)
├── 2. B0 Tabular Baseline (LightGBM)
│     └── Frozen baseline: PR-AUC = 0.64914, ROC-AUC = 0.92502
├── 3. Relational Structural Audit & Graph Diagnostics (Stage A)
│     └── 7 candidate relations evaluated for coverage, fragmentation & lift
├── 4. Train-Only Relational Candidate Screening (Stage B)
│     └── Selection of optimal relational substrates without validation snooping
├── 5. Controlled Alternative B1 Benchmarks
│     └── B0 (0.64914) vs B1-card_core_addr1 (0.64401) vs B1-card1 (0.65504)
├── 6. G1 Graph Neural Network Neighborhood Embeddings
│     └── GraphSAGE embedding + B0 predictors: PR-AUC = 0.62642, below both baselines
├── 7. G1 Attribution Controls (dilution, readout, cross-fitting, budget)
│     └── Verdict: most of the regression was an artifact; corrected G1 reaches B0, not B1
└── 8. Seed Variance Across Training Runs (5 seeds x B0 and B1-card1)
      └── Gain direction holds; PR-AUC magnitude is confounded by the estimator cap
```

---

## 2. Experimental Discipline & Invariance Guarantees

To ensure valid scientific attribution, all experiments adhere to strict data-mining hygiene:

1. **Strict Temporal Split Protocol**:
   - Total rows: **590,540** sorted chronologically by `TransactionDT`.
   - **Train**: 413,378 rows (70.0%, 14,538 frauds, prevalence 3.517%).
   - **Validation**: 88,581 rows (15.0%, 3,042 frauds, prevalence 3.434%).
   - **Test**: 88,581 rows (15.0%, strictly isolated, `test_evaluated = False`).
2. **Leak-Free Categorical Mappings**:
   - Frequency & category encodings fitted **strictly on the train split**.
   - `__MISSING__` (token 0) and `__UNKNOWN__` (token 1) handle unobserved validation categories deterministically.
3. **Cross-Split Historical Continuity with Block Semantics**:
   - Entity history is continuous across train, validation, and test (not reset at split boundaries).
   - **Strictly-before timestamp blocks**: transactions sharing identical timestamps within an entity observe identical preceding history (zero intra-block leakage).
   - Label-free feature generation: target labels (`isFraud`) are never referenced during feature extraction.
4. **Frozen Model Invariance**:
   - Zero parameter tuning across iterations: LightGBM hyperparameters (learning rate `0.03`, `num_leaves` `64`, `min_child_samples` `50`, colsample `0.8`, subsample `0.8`, `reg_alpha` `0.1`, `reg_lambda` `1.0`, estimator cap `6000`, patience `200`) remain strictly frozen from B0 through all B1 variants. The authoritative values live in `reports/baseline/baseline_metadata.json` (`lightgbm_parameters`) and every B1 run re-validates against them before training.

---

## 3. Data Pipeline & Schema

### Raw Feature Profiling (`reports/data_profile/`)

Every column in both raw files was profiled before any modelling: missingness,
cardinality, constant-column detection, numeric distributions, top categorical
values. No column was entirely missing or constant. More usefully, the profile
separated columns that read as **persistent entity identifiers** from those that
read as transactional attributes.

| Column | Missing % | Distinct | Read as |
| :--- | ---: | ---: | :--- |
| `card1` | 0.0% | 13,553 | Near-complete, identifier-scale cardinality |
| `card2`/`card3`/`card5` | 0.3–1.5% | 114–500 | Card attributes, low missingness |
| `card4`/`card6` | 0.3% | 4 | Network/type — refines rather than defines an entity |
| `addr1`/`addr2` | 11.1% | 332 / 74 | Billing region |
| `DeviceInfo` | 17.7% | 1,786 | Device string |
| `id_30`/`id_31`/`id_33` | 2.7–49% | 75–260 | OS / browser / resolution |

`card1` stood out here — before any structural testing — as the only candidate
identity field with 0.0% missingness at identifier-like cardinality.
`R_emaildomain` (76.8% missing) was excluded as too sparse to build history on,
and `P_emaildomain` was a marginal candidate not pursued in favour of the card
fields.

This step was exploratory and domain-driven rather than a formal selection
procedure. It produced the seven *candidate* entity definitions that Section 4
then tests; it did not answer which of them works.

**A clarification worth stating plainly, because it is easy to misread.**
Profiling informed which columns to build *new relational features from*. It did
not remove anything from the baseline. B0 keeps all 435 raw predictors
regardless of how they profiled.

### Raw Data Ingestion & Preprocessing (`src/data/`)
- Ingests `train_transaction.csv` (590,540 rows, 394 cols) and `train_identity.csv` (144,233 rows, 41 cols).
- Left-joins on `TransactionID` yielding 590,540 rows and 435 tabular predictors (404 numeric, 31 categorical).
- Generates `data/processed/model_dataset.parquet` (snappy compressed) and `data/processed/split_assignment.parquet`.

---

## 4. Stage A: Relational Audit & Graph Structure Analysis

We evaluated **7 candidate entity relations** to determine their topological suitability for fraud detection:

```python
CANDIDATES = {
    "card1": ["card1"],
    "card1_card2": ["card1", "card2"],
    "card_core": ["card1", "card2", "card3", "card5"],
    "card_full": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "card_core_addr1": ["card1", "card2", "card3", "card5", "addr1"],
    "device_info": ["DeviceInfo"],
    "device_fingerprint": ["DeviceInfo", "id_30", "id_31", "id_33"],
}
```

### Structural Screening Summary (`reports/relational_screening/relation_screening.csv`)

| Relation | Structural Classification | Coverage (%) | Largest Entity Share (%) | Largest Component (%) | Median Group Size | Recurring Entity (%) | Median Repeat Gap (h) | Fraud Neighbor Lift | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`card1`** | **Promising** | **100.0%** | 2.45% | 2.45% | 4.0 | 73.05% | 1.51 | 8.49 | **Preferred** |
| **`card1_card2`** | **Promising** | **98.42%** | 2.49% | 2.45% | 4.0 | 73.02% | 1.50 | 8.40 | **Preferred** |
| **`card_core`** | **Promising** | 98.02% | 2.38% | 2.33% | 4.0 | 72.90% | 1.51 | 8.40 | **Shortlist** |
| **`card_full`** | **Promising** | 98.01% | 2.38% | 2.33% | 4.0 | 72.89% | 1.51 | 8.40 | **Shortlist** |
| **`card_core_addr1`** | **Promising** | 86.67% | 1.14% | 0.99% | 2.0 | 58.86% | 12.12 | 10.77 | **Shortlist** |
| **`device_info`** | **Problematic** | 22.10% | 40.13% | 8.87% | 4.0 | 74.19% | 0.05 | 12.77 | **Reject** |
| **`device_fingerprint`** | **Problematic** | 13.58% | 4.55% | 0.62% | 2.0 | 57.77% | 1.63 | 13.21 | **Reject** |

### Univariate Signal Measurement

Single-feature discrimination is reported at two tiers, because the two answer different questions:

| Tier | Rows scored | What it measures |
| :--- | :--- | :--- |
| **All-rows** (`pr_auc`, `roc_auc_ascending`) | Every training row; a NaN recency is imputed to a sentinel above the observed maximum | Comparable across a relation's four features — they share a row set and a base rate. Includes any missingness effect. |
| **Covered-rows** (`pr_auc_covered`, `roc_auc_covered_ascending`) | Only rows whose grouping key is present *and* whose feature is observed | The entity-history signal with the missingness confound removed. Candidate selection ranks on this. |

Two further conventions keep the numbers honest:

- **ROC-AUC is always reported in the raw ascending direction**, so an inverse association shows as a value below `0.5` rather than being silently flipped. PR-AUC is reported for the stronger direction with `direction` naming which one.
- **PR-AUC is reported as a lift over its own base rate** (`pr_auc_lift`, `pr_auc_lift_covered`). At a 3.5% fraud rate a random scorer already achieves PR-AUC ≈ 0.035, so absolute cutoffs would label every summary "weak" by construction.

### Promotion Gates

A relation is labelled `preferred` only if it clears every gate below — no single scalar can promote it:

| Gate | Requirement |
| :--- | :--- |
| `structural_classification_is_promising` | Passed all Stage A structural thresholds |
| `coverage_at_least_preferred_minimum` | Coverage ≥ 80% (stricter than the 50% structural floor) |
| `largest_entity_below_preferred_maximum` | Largest entity < 5% of covered volume |
| `history_signal_at_least_moderate` | Best covered-rows PR-AUC lift ≥ 1.25× |
| `history_signal_exceeds_key_missingness` | Best covered-rows lift > the lift of a bare "key is missing" flag |

Qualifiers are then ranked by `coverage × (covered lift − 1)` and at most **two** are promoted, because Stage B permits at most two controlled validation experiments.

### Rejection Rationale
- **`device_info`**: Catastrophic super-entity collapse — a single entity accounts for 40.13% of all covered volume (violating the <10% threshold).
- **`device_fingerprint`**: Severe missingness — only covers 13.58% of transactions (violating the >50% threshold).

---

## 5. Iteration Results & Benchmark Comparison

Four feature summaries were generated per relation:
1. `prior_count`: Lifetime previous transactions in entity.
2. `prior_count_24h`: Transactions in preceding 24 hours (`86,400s`).
3. `prior_count_7d`: Transactions in preceding 7 days (`604,800s`).
4. `time_since_previous_hours`: Elapsed hours since immediate previous transaction (`NaN` if first occurrence).

### Cross-Model Comparison Matrix (`reports/b1/b1_cross_relation_comparison.csv`)

| Model Variant | Entity Definition | Predictors | Validation PR-AUC | Validation ROC-AUC | $\Delta \text{PR-AUC}_{\text{val}}$ vs B0 | 95% CI (paired bootstrap) | Relational Feature Gain Ranks (/439) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`B0` (Baseline)** | Tabular Only | 435 | **0.64914** | **0.92502** | Baseline | — | — |
| **`B1-card_core_addr1`** | Composite Core + Addr1 | 439 | **0.64401** | **0.92348** | $-0.00513$ | not computed | #15, #18, #32, #49 |
| **`B1-card1`** | Card 1 Only | 439 | **0.65504** | **0.92807** | **+0.00590** | $[+0.00199, +0.00976]$ | **#14, #16, #19, #31** |
| **`B1-card1_card2`** | Card 1 + Card 2 | 439 | **0.65465** | **0.92703** | **+0.00551** | $[+0.00161, +0.00936]$ | **#15, #16, #18, #31** |

> **The PR-AUC deltas in this table are single-seed measurements and must not be quoted as point estimates.** Refitting B0 and B1-card1 across five seeds moves the paired delta from $-0.00216$ to $+0.01402$. The movement is an early-stopping artifact rather than a property of the relational features, and the underlying gain is stable once that is controlled for, but the specific figure $+0.00590$ is one draw from a wide distribution. See Section 9 for the panel and the converged figure, and Section 10 for the permuted-entity null that establishes the gain is entity history rather than a split-search artifact.

The CI column is a paired bootstrap over validation rows at the frozen models (`reports/b1/b1_significance.json`, generated by `src.models.compare_b1_significance` from the persisted `validation_predictions.parquet` files — no retraining), computed by the same shared module (`src/models/significance.py`) the G1 significance report uses. Both B1-card1 and B1-card1_card2 exclude zero against B0. Read head-to-head, they do not distinguish from each other: B1-card1 minus B1-card1_card2 is $+0.00039$ with a 95% CI of $[-0.00255, +0.00338]$, which spans zero. card1 is reported as the winning relation on point estimate and simplicity (fewer moving parts, no `card2` merge), not because the bootstrap shows it beating card1_card2.

### Key Scientific Insights
1. **The Initial B1 Regression was Relational Selection Error**:
   - `card_core_addr1` fragmented the entity space (13.33% zero-fallback rate, 12.1h repeat gap), diluting temporal burst signals.
2. **Its Apparent Univariate Signal was Missingness, Not History**:
   - `card_core_addr1_prior_count` looks informative on the full training partition (ROC-AUC `0.3817` — a strong *inverse* association). Restricted to rows where the grouping key is actually present, its ROC-AUC is `0.5004`: no signal whatsoever.
   - The association is entirely the zero-count sentinel handed to uncovered rows, which carry a **10.18%** fraud rate against **2.49%** on covered rows. A bare "key is missing" flag scores PR-AUC `0.0609`, *above* the feature itself.
   - This is why `card_core_addr1` is shortlisted but never preferred: the screening requires a relation's history summaries to beat its own key-missingness indicator. See `reports/relational_screening/coverage_confound.csv`.
3. **High Coverage Restores Strong Positive Signal**:
   - Switching to `card1` (100% coverage) or `card1_card2` (98.42% coverage) yields a clean **+0.01103 PR-AUC recovery** over `card_core_addr1` and outperforms the tabular B0 baseline by **+0.00590 PR-AUC** on the frozen seed (outcome `A` in `reports/b1/b1_cross_relation_summary.json`). The direction of that gain is confirmed across seeds and on ROC-AUC; the magnitude is not — see Section 9. The converged figure is $+0.00630$, and a permuted-entity null control (Section 10) establishes that the gain comes from genuine entity history rather than from widening the split search by four columns.
4. **Feature Salience**:
   - In `B1-card1`, all 4 relational features rank in the **top 7% by gain** (#14, #16, #19, #31 out of 439 total features).
   - That says all four are *used*, not that all four are *needed*. Which of them
     carries the gain is the subject of Section 11, and Section 12 shows the
     answer depends on the reporting protocol and is not currently settled.

---

## 6. Repository Layout

```
fraud-relational-ml/
├── data/
│   ├── raw/                                           # Original IEEE-CIS CSV files
│   └── processed/
│       ├── model_dataset.parquet                      # Frozen B0 tabular dataset (590,540 rows, 435 predictors)
│       ├── split_assignment.parquet                   # Frozen temporal train/val/test splits
│       ├── relational_features_card_core_addr1.parquet# B1 card_core_addr1 features
│       ├── relational_features_card1.parquet          # B1 card1 features
│       └── relational_features_card1_card2.parquet    # B1 card1_card2 features
├── models/
│   ├── lightgbm_baseline.txt                          # Frozen B0 LightGBM model
│   ├── lightgbm_b1_card_core_addr1.txt                # B1 card_core_addr1 model
│   ├── lightgbm_b1_card1.txt                          # B1 card1 model
│   ├── lightgbm_b1_card1_card2.txt                    # B1 card1_card2 model
│   ├── lightgbm_g1_card1.txt                          # G1 model, plus four attribution controls
│   ├── graphsage_card1_encoder*.pt                    # Encoders: base, cross-fitted folds, variants
│   ├── ablation/                                      # Eight ablation cells plus a five-seed panel
│   ├── convergence_check/                             # Extended-cap and alternative-stopping runs
│   ├── seed_variance/                                 # Five-seed B0 and B1-card1 panels
│   └── permuted_null/                                 # Permuted-entity null control runs
├── configs/
│   └── screening.json                                 # Stage A/B screening thresholds (policy, not invariants)
├── reports/
│   ├── baseline/                                      # B0 metrics, metadata, importance
│   ├── data_profile/                                  # Column-level profile of all 435 raw predictors
│   ├── ablation/                                      # Per-feature ablation panel & verdict
│   ├── selection_bias/                                # Argmax exposure screen & equal-budget panel
│   ├── stages/                                        # Cross-stage comparison table & summary
│   ├── operating_points/                              # Calibration, alert budgets, cost sweep
│   ├── ledger/                                        # Experiment ledger over every model artifact
│   ├── relational_audit/                              # Entity & graph diagnostics CSVs
│   ├── relational_screening/                          # Stage A & B screening reports & JSON
│   │   ├── relation_screening.csv                     # Per-relation structural + signal summary
│   │   ├── feature_discrimination.csv                 # All-rows and covered-rows univariate stats
│   │   ├── coverage_confound.csv                      # Signal attributable to key missingness alone
│   │   ├── feature_redundancy.csv                     # Spearman correlations between the 4 features
│   │   └── candidate_selection.json                   # reject / shortlist / preferred + gate results
│   ├── relational_features/                           # Feature builder metadata manifests
│   ├── permuted_null/                                 # Permuted-entity null runs, comparison table & verdict
│   └── b1/
│       ├── card_core_addr1/                           # B1 card_core_addr1 report suite
│       ├── card1/                                     # B1 card1 report suite
│       ├── card1_card2/                               # B1 card1_card2 report suite
│       ├── b1_cross_relation_comparison.csv           # 4-way consolidated metrics table
│       └── b1_cross_relation_summary.json             # Structured outcome & decision rules
├── src/
│   ├── data/
│   │   ├── make_temporal_split.py                     # Temporal split generator
│   │   └── build_model_dataset.py                     # Ingestion & cleaning pipeline
│   ├── graph/
│   │   ├── analyze_relations.py                       # Entity & graph diagnostic audit
│   │   ├── temporal_contract.py                       # Strictly-before admissibility rule
│   │   ├── temporal_sampler.py                        # Target-anchored neighbour sampler
│   │   ├── build_transaction_graph.py                 # card1 node & entity-edge tables
│   │   ├── train_graphsage_encoder.py                 # Frozen G1 GraphSAGE encoder
│   │   └── train_graphsage_variants.py                # G1 attribution-control encoders
│   ├── features/
│   │   ├── build_relational_features.py               # Generalized feature engineer
│   │   └── screen_relations.py                        # Train-only screening module
│   └── models/
│       ├── train_lightgbm_baseline.py                 # Frozen B0 LightGBM trainer
│       ├── train_lightgbm_relational.py               # Parametrized B1 LightGBM trainer
│       ├── train_lightgbm_g1.py                       # Frozen G1 embedding + LightGBM trainer
│       ├── train_lightgbm_g1_controls.py              # G1 attribution-control runs
│       ├── compare_g1_controls.py                     # Control comparison & pre-agreed verdict
│       ├── train_seed_variants.py                     # Seed-variance runs for B0 and B1-card1
│       ├── summarize_seed_variance.py                 # Seed spread & early-stopping stratification
│       ├── train_lightgbm_convergence_check.py        # Extended-cap convergence runs for B0/B1-card1/G1-card1
│       ├── summarize_convergence_check.py             # Capped-vs-converged deltas & the cap decision
│       ├── significance.py                            # Shared paired-bootstrap module (no trainer dependency)
│       ├── compare_b1_significance.py                 # Paired-bootstrap CIs for the three B1 comparisons
│       ├── compare_b1_variants.py                     # Cross-variant comparison generator, consumes the CIs
│       ├── train_lightgbm_permuted_null.py            # Permuted-entity null control runs for card1
│       ├── compare_permuted_null.py                   # Null comparison & the pre-registered verdict rule
│       ├── train_lightgbm_ablation.py                 # Singleton and leave-one-out ablation cells
│       ├── compare_ablation.py                        # Ablation verdict under the pre-registered rule
│       ├── summarize_ablation_seed_panel.py           # Seed panel for a single ablation variant
│       ├── selection_bias.py                          # Argmax / matched / plateau estimators
│       ├── compare_selection_bias.py                  # Exposure screen over every published comparison
│       ├── rederive_ablation_at_equal_budget.py       # Equal-budget re-derivation with intervals
│       ├── compare_converged_significance.py          # Paired-bootstrap CIs at the converged protocol
│       ├── compare_stages.py                          # Cross-stage table spanning B0, B1 and G1
│       ├── calibration_and_operating_points.py        # Reliability, alert budgets, cost sweep
│       └── build_experiment_ledger.py                 # Index of every model artifact and its claims
└── tests/
    ├── test_lightgbm_baseline.py                      # B0 unit test suite
    ├── test_lightgbm_relational.py                    # B1 merge & invariant test suite
    ├── test_relational_features.py                    # Feature calculation unit tests
    ├── test_relational_screening.py                   # Stage A & B screening test suite
    ├── test_relational_models.py                      # Multi-model verification suite
    ├── test_temporal_sampler.py                       # Strictly-before sampling correctness
    ├── test_graphsage_encoder.py                      # Encoder & leakage-guard suite
    ├── test_lightgbm_g1.py                            # G1 merge suite
    ├── test_g1_controls.py                            # Attribution-control suite
    ├── test_seed_variance.py                          # Seed-variance & stratification suite
    ├── test_convergence_check.py                      # Convergence-check & cap-decision suite
    ├── test_significance.py                           # Shared paired-bootstrap module suite
    ├── test_compare_b1_significance.py                # B1 significance report suite
    ├── test_compare_b1_variants.py                    # B1 comparison report's CI-consumption suite
    ├── test_permuted_null.py                          # Permutation, reference-pinning & verdict suite
    ├── test_ablation.py                               # Ablation manifest, isolation & verdict suite
    ├── test_ablation_seed_panel.py                    # Ablation seed-panel stratification suite
    ├── test_selection_bias.py                         # Estimator & exposure-screen suite
    ├── test_compare_selection_bias.py                 # Screen classification & noise-floor sourcing
    ├── test_rederive_ablation_at_equal_budget.py      # Equal-budget panel classification suite
    ├── test_compare_stages.py                         # Cross-stage registry & classifier suite
    ├── test_calibration_and_operating_points.py       # Calibration, budget & cost-sweep suite
    └── test_experiment_ledger.py                      # Ledger completeness & disclosure suite
```

The suite is **662 tests**: 661 in the gating run plus one throughput benchmark
that is marked and deselected. Continuous integration runs the gating set on
every push and pull request. On a clean checkout 47 of them skip, because the
raw dataset is gitignored and the tests that need it guard on its presence; the
synthetic-fixture tests that carry the suite run regardless.

---

## 7. Execution & Reproduction Guide

### Obtaining the Raw Data

The raw inputs are not in this repository. They are roughly 1.3 GB, and they are
distributed by Kaggle under the **IEEE-CIS Fraud Detection** competition rules,
which govern access and redistribution — so `data/raw/` is gitignored and this
project redistributes none of it. Download it yourself from the
[competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
after accepting those rules.

Two files are required, and both go in `data/raw/`:

| File | Data rows | SHA-256 | Role |
| :--- | ---: | :--- | :--- |
| `train_transaction.csv` | 590,540 | `3a5c83ab…83d642` | Transactions and the `isFraud` label. Every split here is carved from it. |
| `train_identity.csv` | 144,233 | `b63c725d…03c37c` | Device and identity attributes, left-joined where present. |

**Three files in the download are deliberately unused**, and the distinction is
worth stating because getting it wrong is costly. `test_transaction.csv`,
`test_identity.csv` and `sample_submission.csv` are the *competition's*
unlabelled holdout — they carry no `isFraud` column at all. They are not this
project's test partition, which is carved from the labelled training file by the
temporal split in Section 3.

Verify before running anything:

```bash
python -m src.data.verify_raw_inputs
```

It checks presence, row counts and checksums, and names exactly what is missing
or mismatched rather than failing later inside a parser. Full digests are
recorded in `src/data/verify_raw_inputs.py`. Pass `--skip-checksums` for a fast
presence-and-length check on a 1.3 GB input.

A checksum mismatch does not necessarily mean the data is wrong, but it does mean
the committed reports cannot be expected to reproduce byte for byte — which is
worth knowing before hours of training rather than after.

### Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce Feature Generation & Training

The orchestrator covers the core pipeline: raw-input verification, profiling,
the temporal split, the model dataset, the relational audit, screening, feature
generation and the B1 training and comparison. It fixes the stage ordering in
one place and checks prerequisites before starting, so skipping a step reports
*which stage* is missing rather than which file.

Four of those stages -- profiling, the temporal split, the model dataset and the
relational audit -- were never in this guide, which previously began at
screening and so could not be followed from a clean checkout.

```bash
# Everything from raw inputs to the B1 comparison, in one command.
# Prerequisites are checked before any work starts, and a stage whose outputs
# already exist is skipped unless --force is passed.
python -m src.pipeline --all

# Inspect what the pipeline will do, without running it
python -m src.pipeline --list
python -m src.pipeline --all --dry-run

# Run one stage, or resume from one
python -m src.pipeline --stage screening
python -m src.pipeline --from relational_features
python -m src.pipeline --stage train_b1 --force

# --- Beyond the core pipeline: the investigation stages, run directly. ---
# These are deliberately not orchestrated; each is a separate enquiry rather
# than a step every reproduction must take.

# 1. Build the card1 graph, train the G1 encoder, and run G1
python -m src.graph.build_transaction_graph
python -m src.graph.train_graphsage_encoder
python -m src.models.train_lightgbm_g1

# 2. G1 attribution controls, then the comparison and pre-agreed verdict
python -m src.graph.train_graphsage_variants --skip-existing
python -m src.models.train_lightgbm_g1_controls --skip-existing
python -m src.models.compare_g1_controls

# 3. Seed-variance panel, then its report
python -m src.models.train_seed_variants --skip-existing
python -m src.models.summarize_seed_variance

# 4. Estimator-cap convergence check, then its report
python -m src.models.train_lightgbm_convergence_check --skip-existing
python -m src.models.train_lightgbm_convergence_check --config b0 --config b1_card1 --stop-metric auc --skip-existing
python -m src.models.summarize_convergence_check

# 5. Permuted-entity null control for the B1-card1 gain, then its verdict
python -m src.models.train_lightgbm_permuted_null --skip-existing
python -m src.models.compare_permuted_null

# 6. Per-feature ablation of the four card1 summaries, then its verdict
python -m src.models.train_lightgbm_ablation --skip-existing
python -m src.models.compare_ablation
python -m src.models.summarize_ablation_seed_panel

# 7. Selection-bias screen and the equal-budget re-derivation (no refits)
python -m src.models.compare_selection_bias
python -m src.models.rederive_ablation_at_equal_budget

# 8. Converged-protocol intervals, then the cross-stage comparison
python -m src.models.compare_converged_significance
python -m src.models.compare_stages

# 9. Calibration and alert-budget operating points (no refits)
python -m src.models.calibration_and_operating_points

# 10. Experiment ledger over every model artifact
python -m src.models.build_experiment_ledger

# 11. Run the gating test suite
python -m pytest -m "not benchmark" -q
```

Step 15 excludes benchmarks deliberately. The suite contains one throughput
measurement that asserts on wall-clock time; it is real information but it fails
under machine load, so it is marked `benchmark` and kept out of the gating run.
`python -m pytest -m benchmark` runs it on its own.

Steps 11, 13 and 14 retrain nothing. They read persisted learning curves,
validation predictions and metadata manifests, so they complete in seconds to
minutes rather than hours.

Both control runners accept `--skip-existing`, so an interrupted sweep resumes
rather than recomputing blocks that are already published. The encoder variants
each take tens of minutes on CPU; the frozen B0, B1 and G1 artifacts are
hash-verified before and after every control run and are never rewritten.

The seed-variance panel is ten full LightGBM fits at the 6,000-round cap and
takes roughly four hours on twelve cores, with a peak resident set near 4.5 GB
per run. Runs are ordered seed-major, so an interruption leaves complete paired
deltas for the seeds that finished rather than a half-built configuration. To
run a single cell, pass `--config` and `--seed`:

```bash
python -m src.models.train_seed_variants --config b0 --seed 202
```

The convergence check is three full LightGBM fits at 15,000 rounds (2.5x the
frozen cap) for the required `average_precision`-patience run, plus two more
at the same cap with patience reordered onto `auc` for B0 and B1-card1. Peak
resident set is comparable to the seed-variance panel's per-run figure; close
other memory-heavy applications before running, and run one process at a
time. `--skip-existing` resumes an interrupted job without recomputing a
completed `(config, stop_metric, seed)` cell:

```bash
python -m src.models.train_lightgbm_convergence_check --config g1_card1 --stop-metric average_precision
```

The permuted-entity null is three full LightGBM fits at the 15,000 cap, one
per permutation seed, each rebuilding the four card1 features on its own
permutation first. Observed wall time is 18 to 45 minutes per seed depending
on where early stopping lands, at a peak resident set comparable to the other
panels, so run one process at a time. `--skip-existing` resumes an interrupted
panel, and a single seed can be run on its own:

```bash
python -m src.models.train_lightgbm_permuted_null --permutation-seed 3 --skip-existing
```

---

## 8. Stage G1: Graph Neural Network Relational Embeddings

With `card1` established as the relational substrate, G1 replaced B1's four scalar summaries with a learned representation: a 2-hop inductive GraphSAGE encoder (fan-out 10x10, mean aggregator, 32-dimensional embedding) trained on `isFraud` through a disposable classification head, feeding its per-transaction embedding into the frozen B0 LightGBM configuration.

Neighbourhoods are drawn under the strictly-before temporal contract (`src/graph/temporal_contract.py`): the target's own timestamp bounds every hop, so no transaction can aggregate information from transactions that had not yet happened.

### First Result

| Model Variant | Predictors | Validation PR-AUC | Validation ROC-AUC | $\Delta \text{PR-AUC}_{\text{val}}$ vs B0 |
| :--- | :--- | :--- | :--- | :--- |
| **`B0` (Baseline)** | 435 | **0.64914** | **0.92502** | Baseline |
| **`B1-card1`** | 439 | **0.65504** | **0.92807** | **+0.00590** |
| **`G1-card1`** | 467 | **0.62642** | **0.91386** | $-0.02272$ |

G1 scored below both, with paired bootstrap intervals excluding zero against each. That established the size of the gap but not its cause. Three properties of the run each offered an explanation having nothing to do with graph structure being unhelpful:

1. **The embedding was not a relational feature block.** Encoder input was the same 435 raw predictors LightGBM already receives, and the standard GraphSAGE readout concatenates the target's own transformed features onto the aggregated neighbourhood, so the block partly re-encoded signal the model already held.
2. **The encoder was fit on the rows the downstream model trains on, without cross-fitting.** Train-row embeddings carry information from those rows' own labels; validation-row embeddings do not.
3. **The encoder was barely trained.** Best epoch 4, roughly 0.40 passes over the train partition, early-stopped on a 15,000-row validation subsample.

### Attribution Controls (`reports/g1_controls/`)

Four controls separate those confounds from the graph verdict. The three encoder controls share one raised training budget, so each is read against `extended_budget` rather than against the frozen run, and only one thing changes at a time. Every control is measured by paired bootstrap against B0, B1-card1 **and** the frozen G1, and all three references are hash-pinned before and after each run.

| Control | Isolates | Validation PR-AUC | $\Delta$ vs B0 (95% CI) | $\Delta$ vs B1-card1 (95% CI) | $\Delta$ vs G1 (95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `shuffled_embedding` | Split-search dilution | 0.62957 | $-0.01957$ $[-0.0247, -0.0145]$ | $-0.02547$ $[-0.0308, -0.0201]$ | $+0.00315$ $[-0.0024, +0.0087]$ |
| `extended_budget` | Training budget, monitor size | 0.61485 | $-0.03429$ $[-0.0407, -0.0278]$ | $-0.04019$ $[-0.0468, -0.0337]$ | $-0.01157$ $[-0.0167, -0.0064]$ |
| `neighbourhood_only` | Self-contribution in readout | **0.64494** | $-0.00420$ $[-0.0095, +0.0010]$ | $-0.01010$ $[-0.0155, -0.0048]$ | $+0.01852$ $[+0.0123, +0.0247]$ |
| `cross_fitted` | Label information in train rows | 0.57948 | $-0.06966$ $[-0.0767, -0.0627]$ | $-0.07556$ $[-0.0828, -0.0683]$ | $-0.04694$ $[-0.0540, -0.0398]$ |

### Key Scientific Insights

1. **Most of the Original Regression was Split-Search Dilution, Not the Graph**:
   - A randomly-aligned 32-column block, holding the real block's per-split marginals and destroying only its row alignment, costs $-0.01957$ PR-AUC against B0 on its own.
   - That accounts for **86.1%** of G1's $-0.02272$ gap, and the shuffled null is statistically **indistinguishable from the real G1 block** ($+0.00315$, CI spans zero). The frozen embedding added nothing that a noise block of the same width did not.
2. **The Readout was Re-encoding Features the Model Already Had**:
   - Dropping the target's own features from the final readout recovers $+0.01852$ PR-AUC over the frozen G1 and brings the stage to **statistical parity with B0** (CI spans zero).
   - This is the single largest correction, and it confirms that a self-inclusive readout cannot isolate a relational contribution when the downstream model already holds every self feature.
3. **Training the Encoder Harder Made the Pipeline Worse**:
   - Raising the budget from 0.40 to 2.0 passes over the train partition, and selecting on the full validation partition instead of a 15,000-row subsample, *lowered* downstream PR-AUC to 0.61485 -- significantly below the frozen G1 ($-0.01157$, CI excludes zero).
   - The importance tables show why: the more the model leans on the embedding block, the worse it scores. `extended_budget` places **30 of 32** embeddings in the top 50 of 467 features by gain and performs worst; `neighbourhood_only` places 13 and performs best; LightGBM correctly ranks the *shuffled* block lowest of all (10 of 32) and still beats both un-cross-fitted real-embedding runs.
   - This is the signature of a non-transferable feature: high train-side gain, negative validation value. Structurally it is the same trap as `card_core_addr1`'s missingness sentinel in Section 5, one level up.
4. **The Verdict, by a Rule Fixed Before the Runs**:
   - The stopping rule agreed in advance: if cross-fitting **and** the neighbourhood-only readout each bring G1 to at least parity with B1-card1, the regression was an encoder artifact and the stage continues; otherwise the negative result stands as a genuine finding.
   - `neighbourhood_only` reaches B0 parity but remains significantly below B1-card1 ($-0.01010$, CI excludes zero), so the rule resolves to **`negative_result_stands`**: a supervised graph encoder on this relation, corrected for readout and training budget, does not beat four scalar relational summaries. The G1 epic closes and effort returns to hardening the B1 claim.
   - The corrected picture is nonetheless very different from the original one. The claim "learned embeddings actively hurt" is **not** supported by these controls; the defensible claim is that they reach the tabular baseline and no further.

### Known Limitation of the Cross-Fitting Control

`cross_fitted` embeds train rows with the K fold encoders and validation/test rows with a full-train encoder, and each encoder was initialised from a different seed. Nothing constrains independently initialised encoders to agree on a latent basis, so an embedding column denotes a different direction either side of the train/validation boundary. The published diagnostic measures exactly this: a standardised train-vs-validation mean gap of **0.390, with 11 of 32 columns above 0.5**, against 0.123-0.233 and 0-2 columns for every single-encoder block in the table.

Its $-0.07556$ therefore confounds removing label leakage with misaligning the feature block, and is **not** evidence for what cross-fitting alone costs or gains. The verdict does not rest on it: the rule requires *both* verdict controls to reach parity, and `neighbourhood_only` -- a single-encoder block with no such defect -- does not. A corrected run should share one initialisation across the full-train and fold encoders, or align each fold encoder's output to the full-train encoder before assembling the block. The limitation is recorded in `reports/g1_controls/g1_control_summary.json` under `known_limitations`.

---

## 9. Seed Variance and the Estimator Cap (`reports/seed_variance/`)

Every headline figure above came from exactly one training run per configuration, so the only uncertainty ever quantified was sampling variance: the paired bootstrap holds a trained model fixed and resamples validation rows. Run-to-run variance under a different seed was unmeasured. B0 and B1-card1 were therefore refitted across five seeds (42, 202, 707, 1337, 2024) with `random_state` as the only parameter permitted to move; every other setting, the feature manifest and the train-only categorical mappings were read from the frozen artifacts and asserted unchanged before training.

Seed 42 reproduces the frozen B0 (`0.649138574648`) and frozen B1-card1 (`0.655039132862`) to a difference of exactly `0.0`, so the remaining four seeds differ from the published runs by seed and by nothing else.

### The Paired Delta by Seed

| Seed | B0 PR-AUC | B1-card1 PR-AUC | $\Delta$ PR-AUC | $\Delta$ ROC-AUC | B0 iter | B1 iter | Stopped early |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 | 0.64913857 | 0.65503913 | $+0.00590$ | $+0.00305$ | 5,821 | 5,999 | neither |
| 202 | 0.64012433 | 0.65414788 | $+0.01402$ | $+0.00190$ | **3,648** | 5,999 | **B0** |
| 707 | 0.64480743 | 0.64264537 | $-0.00216$ | $+0.00309$ | 5,926 | **3,644** | **B1** |
| 1337 | 0.64279500 | 0.64942275 | $+0.00663$ | $+0.00231$ | 5,999 | 5,994 | neither |
| 2024 | 0.64860778 | 0.64974079 | $+0.00113$ | $+0.00287$ | 5,984 | **4,651** | **B1** |

Across the full panel the paired PR-AUC delta has mean $+0.00511$, standard deviation $0.00615$ and range $[-0.00216, +0.01402]$ — a standard deviation larger than the mean, and a range straddling zero.

### The Instability is an Early-Stopping Artifact

1. **Which model stopped early predicts the delta, in five seeds out of five.**
   - Both models run to the cap (seeds 42, 1337): delta $+0.00590$ and $+0.00663$ — mean $+0.00626$, standard deviation $0.00051$.
   - B0 stops early (seed 202): delta inflates to $+0.01402$.
   - B1 stops early (seeds 707, 2024): delta collapses to $-0.00216$ and $+0.00113$.
   - Contaminated seeds carry a standard deviation of $0.00855$, **17x** the clean stratum's $0.00051$.
2. **The mechanism is that early stopping optimises the reported metric.** Patience is evaluated on validation `average_precision` — the same quantity the comparison reports. A model truncated ~2,300 rounds short of its opponent loses that optimisation, so the PR-AUC difference partly scores the stopping point rather than the predictors.
3. **ROC-AUC, which is not the stopping criterion, is stable and unanimous.** B1-card1 beats B0 on ROC-AUC in **all five seeds**, mean $+0.00264$, standard deviation $0.00052$. This is the cleanest available read of the relational contribution, and it is positive throughout.
4. **The published bootstrap interval understates the true uncertainty by roughly 3x.** Seed standard deviation on the delta is $0.00615$ against the frozen paired bootstrap's $0.00199$ — a ratio of **3.09**. The bootstrap resamples rows but never refits the model, so it cannot see this source at all.

### Consequences

- **The relational gain is real but its PR-AUC magnitude is not currently quotable.** The direction is supported by every ROC-AUC comparison and by both clean PR-AUC comparisons; the figure $+0.00590$ is not a measurement of it.
- **The estimator cap is promoted from a caveat to a blocker.** It is not merely that models are cap-bound rather than converged: the cap-and-patience interaction is what the headline delta partly measures. Note also that early stopping *does* fire — in 3 of these 10 runs — so the previously recorded property that it never triggers is a fact about seed 42, not about the configuration.
- **Any comparison in this repository between two cap-bound LightGBM runs inherits this confound**, including the G1 controls in Section 8.

### Convergence Check at an Extended Cap (`reports/convergence_check/`)

B0, B1-card1 and G1-card1 were each refit once, at seed 42, with the estimator cap raised from 6,000 to 15,000 and the 200-round patience left unchanged; the frozen artifacts were hash-pinned before and after and are untouched. All three now genuinely trigger early stopping — the cap was binding for every run this repository had published:

| Configuration | Frozen best iteration (cap 6,000) | Converged best iteration (cap 15,000) | Gap |
| :--- | :--- | :--- | :--- |
| B0 | 5,821 | 6,011 | +190 |
| B1-card1 | 5,999 | 6,087 | +88 |
| G1-card1 | 5,965 | 7,147 | **+1,182** |

G1-card1's gap is more than 6x either LightGBM configuration's, confirming the suspicion raised when the graph stage closed: 32 additional dense columns needed materially more rounds to converge, and the frozen 6,000-cap comparison caught it mid-fit.

**The B1 gain survives.** Capped delta $+0.00590$ moves to a converged delta of $+0.00630$ (shift $+0.00040$), within twice the seed-variance panel's clean-stratum noise floor ($0.00051$). The direction and approximate magnitude reported throughout this document hold once the cap is no longer a factor.

**The G1 deficit does not.** Capped delta $-0.02272$ moves to a converged delta of $-0.02093$ (shift $+0.00179$) — more than three-and-a-half times the noise floor. The gap between G1-card1 and B0 was measurably inflated by G1 having ~1,200 more rounds' worth of fitting left on the table than B0 did. The converged figure, $-0.0209$, supersedes the capped $-0.0227$ wherever this document quotes the G1 result.

**Decoupling the stopping metric from the reported one was tested and does not work as a fix.** Reordering `eval_metric` so patience watches ROC-AUC instead of `average_precision` was run for B0 and B1-card1 at the same 15,000 cap. Both stopped far earlier than under `average_precision` patience — B0 at iteration 1,121, B1-card1 at 1,318 — because ROC-AUC plateaus on this problem thousands of rounds before PR-AUC does. Read at those rounds, PR-AUC is far below either model's converged value (B0: $0.6013$ vs. $0.6492$ converged; B1-card1: $0.6094$ vs. $0.6555$ converged). The resulting delta ($+0.00807$) is coincidentally still positive and of similar order, but neither model is meaningfully fit at the round it was read from, so this is not a comparison worth trusting. The mechanism this document originally worried about — patience watching the reported metric — turns out not to be the actual defect; the defect was simply that 6,000 rounds was not enough, and average_precision patience correctly detects genuine convergence once the cap is high enough to let it.

**Decision: the capped headline numbers require revision, not just a documented caveat.** Every configuration converges before 15,000 rounds under the original stopping rule, so the cap is not permanently unresolvable — but the specific 6,000-round cap materially understated the G1 deficit. Going forward, any new B0/B1/G1-family training should use a cap of at least 15,000 with patience unchanged on `average_precision`; decoupling the stopping metric is rejected based on the evidence above. The frozen B0, B1-card1 and G1-card1 artifacts remain frozen per this investigation's constraint — refreezing them at a higher cap is a separate, deliberately agreed action, not a side effect of this check.

---

## 10. Permuted-Entity Null Control (`reports/permuted_null/`)

The B1-card1 gain survives sampling noise (the paired bootstrap), refit noise (the seed panel) and convergence (the extended cap). One explanation remained untested. Adding four numeric columns changes LightGBM's split search, and over thousands of boosting rounds that alone can move validation PR-AUC, regardless of what the columns contain. Section 8 is this repository's own evidence that the failure mode is real: a randomly-aligned 32-column block cost $-0.01957$ against B0 on its own, accounting for 86% of the G1 regression.

This control asks the same question of B1 at four columns. `card1` is permuted **globally** across all 590,540 transactions, breaking the correspondence between a transaction and its entity; the same four features are then rebuilt on that permutation by `build_relational_features.py`, unchanged. The permutation preserves the exact multiset of `card1` values — hence the exact per-entity transaction-count distribution — and leaves `TransactionDT` untouched for every row, hence the exact timestamp distribution. Only the assignment of a transaction to an entity is randomised.

The permutation is global rather than within-split, unlike G1's shuffled-embedding null. That control shuffles within each split because the encoder is fit on train rows and a global shuffle would move train-fitted rows across a split boundary. `card1` has no fitting step — it is a raw identifier — and the feature builder already treats entity history as continuous across train, validation and test by design.

Three permutation seeds were run. LightGBM's own `random_state` stays pinned at 42, so the only thing varying between these runs and the converged B1-card1 reference is which entity assignment the four columns were built on. Every run is scored against the **converged** (cap 15,000, `average_precision` patience, seed 42) references established in Section 9, never the frozen cap-6,000 artifacts.

### Results (`reports/permuted_null/permuted_null_comparison.csv`)

| Run | PR-AUC | $\Delta$ vs B0 | 95% CI | $\Delta$ vs B1-card1 | 95% CI | Best iter | Share of real gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B0 (converged) | 0.64919 | — | — | $-0.00630$ | — | 6,011 | — |
| **B1-card1** (converged) | **0.65550** | $+0.00630$ | — | — | — | 6,087 | 100% |
| Permuted, seed 1 | 0.64923 | $+0.00004$ | $[-0.00360, +0.00360]$ | $-0.00627$ | $[-0.01040, -0.00210]$ | 9,390 | $+0.6\%$ |
| Permuted, seed 2 | 0.64323 | $-0.00597$ | $[-0.00959, -0.00239]$ | $-0.01227$ | $[-0.01646, -0.00820]$ | 5,617 | $-94.7\%$ |
| Permuted, seed 3 | 0.64576 | $-0.00343$ | $[-0.00710, +0.00016]$ | $-0.00973$ | $[-0.01380, -0.00572]$ | 8,644 | $-54.4\%$ |

All three permuted runs trigger early stopping before the 15,000 cap, so none is read mid-fit.

### Verdict

The interpretation was fixed before any permuted run was scored: *"recovers a meaningful share of the gain"* means a permuted variant **significantly** beats B0 — its paired-bootstrap 95% CI on the permuted-minus-B0 delta excludes zero on the positive side. This is the same significance test every other comparison in this repository is settled by, and no recovered-fraction threshold is applied; the fraction is reported for magnitude only.

**No permuted variant beats B0.** Outcome: `PERMUTED_AT_OR_BELOW_B0`.

1. **Two of three land significantly *below* B0.** Seeds 2 and 3 recover $-94.7\%$ and $-54.4\%$ of the gain: destroying the entity assignment does not merely remove the benefit, it makes the four columns actively harmful, which is what four columns of structured noise should do.
2. **All three land significantly below B1-card1**, every CI on that comparison excluding zero. The real feature block beats its own permuted counterpart on every seed.
3. **Seed 1 is indistinguishable from B0.** Its $+0.00004$ point estimate is a thousandth the size of the real gain and its interval spans zero. Recorded rather than smoothed over: the summary's `all_permuted_variants_at_or_below_b0` field is therefore `false`, because that field tests the sign of the point estimate. The verdict does not rest on it — it rests on significance, and no interval excludes zero on the positive side.

**This is the mirror image of the G1 result.** There, a random block of 32 columns reproduced almost all of the apparent effect, so the effect was the block's *width*. Here, a random block of 4 columns reproduces none of it, so the effect is the block's *content*. The same instrument, pointed at both results, separates them cleanly.

With sampling noise, refit noise, convergence and structural dilution all ruled out, the B1-card1 gain is attributable to genuine `card1` entity history. The claim about the feature content is now causal, and the magnitude remains small: $+0.00630$ PR-AUC at convergence.

---

## 11. Per-Feature Ablation of the card1 Summaries (`reports/ablation/`)

The relational gain is real, but B1-card1 adds four columns at once and the
headline number is silent on which of them does the work. The four are Spearman
correlated $0.83$–$0.95$ among the counts and $-0.59$–$-0.75$ between counts and
recency, which describes roughly one or two independent factors rather than four.

Eight runs at cap 15,000, scored against the converged references: four
singletons (B0 plus exactly one summary) and four leave-one-out variants
(B1-card1 minus exactly one). Both directions are needed. Leave-one-out alone
would return four nulls, because each column is reconstructible from the others.
Singletons alone cannot distinguish four views of one factor from four additive
signals.

The interpretation rule was fixed before any of the eight ran: a feature
**carries the gain** if its singleton interval excludes zero on the positive
side, and is **non-redundant** if its leave-one-out interval excludes zero on the
negative side.

| Variant | Δ PR-AUC | 95% CI | Excludes zero |
| :--- | ---: | :--- | :--- |
| B0 + `prior_count` | $+0.00931$ | $[+0.00571, +0.01286]$ | yes |
| B0 + `prior_count_7d` | $+0.00851$ | $[+0.00527, +0.01183]$ | yes |
| B0 + `prior_count_24h` | $+0.00488$ | $[+0.00176, +0.00801]$ | yes |
| B0 + `time_since_previous_hours` | $+0.00242$ | $[-0.00067, +0.00556]$ | no |
| B1-card1 − `prior_count` | $+0.00055$ | $[-0.00268, +0.00372]$ | no |
| B1-card1 − `prior_count_24h` | $+0.00548$ | $[+0.00243, +0.00851]$ | yes |
| B1-card1 − `prior_count_7d` | $-0.00099$ | $[-0.00406, +0.00202]$ | no |
| B1-card1 − `time_since_previous_hours` | $+0.00042$ | $[-0.00284, +0.00363]$ | no |

Read under the fixed rule, this is `REDUNDANT_SUMMARIES_ANY_ONE_SUFFICES`: three
features carry the gain alone, none is non-redundant. Recency is the odd one out
in the expected direction — `time_since_previous_hours` is the only summary that
fails to carry the gain on its own, so the negative count/recency correlation
does *not* mark two genuine factors.

**This verdict is contested by Section 12 and should not be quoted as settled.**

---

## 12. Reported PR-AUC Is a Validation Argmax (`reports/selection_bias/`)

Training uses patience 200 on `average_precision`, and the reported PR-AUC is
that curve's maximum. A run continues *precisely because* its curve keeps setting
new maxima, so an arm that plateaus slowly is granted more rounds **and** draws
its reported maximum from more samples of a noisy statistic. Two arms of a
comparison are then not scored under the same amount of selection.

This is not the estimator-cap defect of Section 9. It is present at a cap no run
reaches, where early stopping fires cleanly for both arms. It is a property of
reporting an argmax.

Three estimators of the same paired delta are defined in
`src/models/selection_bias.py`: the **argmax** the pipeline reports, the argmax
over a **matched** budget, and the **plateau** level over a common tail window.
Measured on the leave-one-out `prior_count_24h` cell across five seeds:

| Estimator | Mean | Sign across seeds |
| :--- | ---: | :--- |
| Reported argmax delta | $+0.00567$ | $+++++$ |
| Argmax over a matched budget | $+0.00099$ | $+-++-$ |
| Plateau level, common tail | $+0.00065$ | $+--+-$ |
| ROC-AUC at the *same* selected iteration | $-0.00192$ | $-----$ |

The correlation between extra rounds granted and reported gain is $0.774$. The
last row is decisive: at the same iteration the variant is *worse* on a metric
the stopping rule does not watch, in five seeds out of five and roughly $6.8$
standard deviations from zero — more stable than the PR-AUC benefit it
contradicts. Extra rounds that bought real learning would not degrade ROC-AUC.

### Which comparisons are exposed

Exposure tracks the gap in curve length between the two arms, which makes it a
cheap prospective screen.

| Comparison | Round gap | Argmax | Matched | Classification |
| :--- | ---: | ---: | ---: | :--- |
| B1-card1 vs B0 | 76 | $+0.00630$ | $+0.00630$ | **unexposed** |
| B0 + `prior_count_24h` vs B0 | 144 | $+0.00488$ | $+0.00488$ | **unexposed** |
| G1-card1 vs B0 | 1,136 | $-0.02093$ | $-0.02250$ | direction holds |
| B0 + `prior_count_7d` vs B0 | 3,126 | $+0.00851$ | $+0.00156$ | direction holds |
| B1-card1 − `prior_count` | 2,126 | $+0.00055$ | $-0.00380$ | **in doubt** |
| B1-card1 − `time_since_previous_hours` | 1,616 | $+0.00042$ | $-0.00355$ | **in doubt** |

The two comparisons whose arms stopped within 150 rounds are exactly the two
needing no correction, and they include the headline gain.

### The localisation is undetermined

Re-scoring every model in the ablation panel at one identical budget of 6,011
trees — reachable by inference alone, since each booster was saved truncated at
its own best iteration — and applying the *same* fixed rule gives a different
answer: carriers shrink to `prior_count` and `prior_count_24h`, and
`prior_count`, `prior_count_7d` and `time_since_previous_hours` all become
non-redundant. That is `ADDITIVE_CONTRIBUTIONS`, not redundancy.

Neither answer can be accepted. The argmax protocol rewards whichever arm was
granted more rounds; the equal-budget protocol truncates that same arm. In this
panel that arm is almost always the ablation variant, so the two protocols are
biased in **opposite directions** and bracket the truth. Settling it requires
refits at a budget fixed in advance.

What is *not* in dispute: the $+0.00630$ headline gain (arms 76 rounds apart,
identical under every estimator), `prior_count_24h` as a carrier, and the G1
deficit, which widens under correction and so holds a fortiori.

---

## 13. Cross-Stage Comparison (`reports/stages/`)

`src/models/compare_stages.py` spans B0, the three B1 variants and G1 in one
table. Stages are registered as data carrying their own order, and the outcome
classifier reads stage order rather than any stage name, so a future stage is a
registration rather than an edit to the classifier.

Protocol is explicit rather than assumed, because the frozen artifacts sit at
the 6,000-estimator cap Section 9 found binding — and frozen G1 is the worst
case, with `early_stopping_triggered` false and `estimator_cap_reached` true. It
stopped because it ran out of budget, not because it converged. Rows therefore
declare their protocol and are only ever compared within one.

Converged-protocol intervals cost no refitting, since every converged run
persisted its validation predictions:

| Comparison | Δ PR-AUC | 95% CI | Excludes zero |
| :--- | ---: | :--- | :--- |
| B1-card1 vs B0 | $+0.006304$ | $[+0.002387, +0.010179]$ | yes |
| G1-card1 vs B0 | $-0.020925$ | $[-0.026316, -0.015395]$ | yes |
| G1-card1 vs B1-card1 | $-0.027230$ | $[-0.032748, -0.021658]$ | yes |

Both protocols return the same verdict: **the graph stage fails to beat the
tabular baseline**, frozen $-0.02272$ and converged $-0.02093$, intervals
excluding zero either way. That conclusion no longer depends on which cap it is
read at.

---

## 14. Calibration and Alert-Budget Operating Points (`reports/operating_points/`)

Evaluation elsewhere in this project is ranking-only. Two things decide whether a
model is *usable* rather than merely better, and both are measured here without
retraining anything.

### Calibration, in the opposite direction to the obvious expectation

Every model is trained with `scale_pos_weight` at $27.43$. That should leave
scores inflated. It does not: mean predicted score sits **below** the base rate
for every variant.

| Variant | Prevalence | Mean score | Ratio | ECE | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.0343 | 0.0221 | 0.64 | 0.0135 | 0.0188 |
| B1-card1 | 0.0343 | 0.0220 | 0.64 | 0.0139 | 0.0189 |
| G1-card1 | 0.0343 | 0.0230 | 0.67 | 0.0136 | 0.0195 |

The reliability curve locates it: 84,689 of 88,581 rows — 96% — fall in the
lowest bin, where the model predicts $0.0023$ against an observed fraud rate of
$0.0125$, under-predicting fivefold and dominating the mean. Over-confidence
appears only in the sparse top bins. **A threshold set from the score scale on
the assumption of inflation would be wrong in the unexpected direction.**

Post-hoc recalibration is reported as a diagnostic only. Isotonic fit on
validation and scored on validation reaches an expected calibration error of
*exactly* zero — an artifact, not a result, since a step function fit on its own
evaluation rows can always do that. Platt cannot fit the shape and makes Brier
worse than the raw scores. Neither is a deployment number.

### Where the relational gain actually sits

The validation window spans 31.41 days at roughly 2,820 transactions per day, so
an alert budget can be expressed as a rate.

| Alerts/day | Alerts | Precision | Recall | Δ recall vs B0 | Extra frauds caught |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 314 | 0.971 | 0.100 | $+0.0003$ | $+1$ |
| 25 | 785 | 0.952 | 0.246 | $-0.0003$ | $-1$ |
| 100 | 3,141 | 0.595 | 0.614 | $+0.0023$ | $+7$ |
| 200 | 6,282 | 0.361 | 0.746 | $+0.0066$ | $+20$ |
| **400** | **12,564** | **0.206** | **0.849** | $+0.0151$ | $+46$ |
| 800 | 25,128 | 0.110 | 0.912 | $+0.0046$ | $+14$ |
| 1600 | 50,256 | 0.058 | 0.965 | $+0.0016$ | $+5$ |

The gain is concentrated, not uniform. It peaks near 400 alerts per day and
decays either side; at 25 alerts per day B1-card1 catches *one fewer* fraud than
B0. A team operating at that budget would gain nothing from this feature work.
Quoting "$+0.00630$ PR-AUC" conceals that entirely, which is why the operating
point matters more than the integrated metric.

Cost figures in `cost_sweep.csv` are swept across ratios rather than fixed, carry
the ratio that produced them on every row, and are **per incident, not
amount-weighted** — the persisted predictions carry no transaction amount, so a
missed small fraud and a missed large one count identically. That is the first
limitation to fix before these numbers inform a capacity decision.

---

## 15. Statistical Validation

Every delta in this project is settled by the same paired bootstrap, implemented
once in `src/models/significance.py` and reused by every comparison:

- Validation row indices are resampled with replacement, 10,000 draws.
- The **same** indices are applied to both score vectors, so the comparison is
  paired rather than two independent bootstraps.
- The 95% interval is the empirical percentile range of the resampled delta.
- Seed 42 throughout, so an interval is reproducible from the committed
  prediction files without retraining.

The module reads persisted `validation_predictions.parquet` files and has no
trainer dependency, which is what allowed the converged-protocol intervals in
Section 13 and the equal-budget intervals in Section 12 to be produced without
refitting anything.

**One caveat that applies to every interval here.** The bootstrap measures
sampling noise on the validation rows. It does not measure refit noise, and
Section 9 established that refit noise is the larger of the two for these
effects. An interval excluding zero means the difference is not an artifact of
*which rows* were scored; it does not by itself mean the difference survives
retraining.

---

## 16. Experiment Ledger (`reports/ledger/`)

Fifty-three model artifacts now sit under `models/`. The ledger indexes every one
of them with its stage, role, purpose, the claims resting on it, its limitations
and — explicitly — what it must not be used for.

Facts are read from each run's own manifests at generation time rather than
transcribed, so the ledger cannot drift from the artifacts it describes. Only the
editorial layer is authored. Panel runs are matched by family pattern, and an
artifact matching no family raises rather than being silently omitted, which is
what makes coverage a property rather than a claim.

Stated on every entry: these are research models trained on a 2019 competition
dataset, selected on validation, with output scores that are not probabilities.
**None is a deployable fraud model**, and none has been evaluated on the held-out
test split.

---

## 17. Known Limitations

1. **The cross-fitting control is defective.** Recorded rather than hidden in
   Section 8: the folds were trained from different encoder initialisations, so
   the control confounds cross-fitting with initialisation variance.
2. **The frozen artifacts are not at convergence.** Deliberate — they are kept at
   the 6,000-round cap so published numbers remain reproducible — but it means
   frozen figures understate their models, and frozen G1 most of all. Quote the
   converged deficit of $-0.02093$, never the frozen $-0.02272$.
3. **Seed variance exceeds the published bootstrap intervals.** The clean-stratum
   estimate for the B1 gain is $+0.00626 \pm 0.00051$; the magnitude is not
   quotable to three decimals from a single run.
4. **The within-block localisation is undetermined**, per Section 12. Two
   protocols give two verdicts and bracket the answer.
5. **The encoder budget never converged.** The extended-budget control improved
   on the original encoder without plateauing, so the G1 encoder was still
   improving when training stopped.
6. **The test partition has never been read.** Every number in this document is a
   validation number.
7. **Residual floating-point non-determinism** means metrics may differ in the
   last decimal places across machines.

---

## 18. Next Phase

The measurement-integrity questions are closed. B0, B1-card1 and G1-card1 all
converge before 15,000 rounds; run-to-run variance is measured and stratified;
the permuted-entity null rules out the last structural explanation for the
relational gain; and the estimator itself has now been screened for selection
bias.

The B1-card1 result stands as the project's one positive finding: $+0.00630$
PR-AUC at convergence, attributable to genuine entity history, unexposed to the
argmax bias, and concentrated near a budget of 400 alerts per day.

Remaining work, in priority order:

1. **Settle the ablation localisation** with refits at a budget fixed in advance.
   Two protocols currently disagree and bracket the answer; no further reading of
   existing artifacts will decide it. Eight sequential fits.
2. **Corrected cross-fitting control** with a shared encoder initialisation,
   repairing the defect in Section 8. Independent of the B1 line.
3. **One-shot final test protocol**, once the localisation settles. The test
   partition is read exactly once, under a protocol written before it is opened.
4. **Combined multi-relation variant.** Low priority: `card1` and `card1_card2`
   are not distinguishable on the paired bootstrap, so little is expected.
