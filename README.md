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
│   └── lightgbm_b1_card1_card2.txt                    # B1 card1_card2 model
├── reports/
│   ├── baseline/                                      # B0 metrics, metadata, importance
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
│       ├── train_lightgbm_permuted_null.py           # Permuted-entity null control runs for card1
│       └── compare_permuted_null.py                   # Null comparison & the pre-registered verdict rule
└── tests/
    ├── test_lightgbm_baseline.py                      # B0 unit test suite
    ├── test_lightgbm_relational.py                    # B1 merge & invariant test suite
    ├── test_relational_features.py                    # Feature calculation unit tests
    ├── test_relational_screening.py                   # Stage A & B screening test suite (40 tests)
    ├── test_relational_models.py                      # Multi-model verification suite (62 tests)
    ├── test_temporal_sampler.py                       # Strictly-before sampling correctness
    ├── test_graphsage_encoder.py                      # Encoder & leakage-guard suite
    ├── test_lightgbm_g1.py                            # G1 merge suite
    ├── test_g1_controls.py                            # Attribution-control suite (49 tests)
    ├── test_seed_variance.py                          # Seed-variance & stratification suite (39 tests)
    ├── test_convergence_check.py                      # Convergence-check & cap-decision suite (25 tests)
    ├── test_significance.py                           # Shared paired-bootstrap module suite
    ├── test_compare_b1_significance.py                # B1 significance report suite
    ├── test_compare_b1_variants.py                    # B1 comparison report's CI-consumption suite
    └── test_permuted_null.py                          # Permutation, reference-pinning & verdict suite (41 tests)
```

---

## 7. Execution & Reproduction Guide

### Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce Feature Generation & Training
```bash
# 1. Train-only relation screening (Stage A; no validation is read)
python -m src.features.screen_relations

# 2. Build relational features for the relations it prefers
python -m src.features.build_relational_features --relation card1
python -m src.features.build_relational_features --relation card1_card2

# 3. Train B1 models
python -m src.models.train_lightgbm_relational --relation card1
python -m src.models.train_lightgbm_relational --relation card1_card2

# 4. Paired-bootstrap CIs for the B1 comparisons, then the comparison report
python -m src.models.compare_b1_significance
python -m src.models.compare_b1_variants

# 5. Build the card1 graph, train the G1 encoder, and run G1
python -m src.graph.build_transaction_graph
python -m src.graph.train_graphsage_encoder
python -m src.models.train_lightgbm_g1

# 6. G1 attribution controls, then the comparison and pre-agreed verdict
python -m src.graph.train_graphsage_variants --skip-existing
python -m src.models.train_lightgbm_g1_controls --skip-existing
python -m src.models.compare_g1_controls

# 7. Seed-variance panel, then its report
python -m src.models.train_seed_variants --skip-existing
python -m src.models.summarize_seed_variance

# 8. Estimator-cap convergence check, then its report
python -m src.models.train_lightgbm_convergence_check --skip-existing
python -m src.models.train_lightgbm_convergence_check --config b0 --config b1_card1 --stop-metric auc --skip-existing
python -m src.models.summarize_convergence_check

# 9. Permuted-entity null control for the B1-card1 gain, then its verdict
python -m src.models.train_lightgbm_permuted_null --skip-existing
python -m src.models.compare_permuted_null

# 10. Run complete test suite
python -m pytest tests/ -v
```

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

## 11. Next Phase

The measurement-integrity questions are closed. B0, B1-card1 and G1-card1 all converge before 15,000 rounds under the existing stopping rule; run-to-run variance is measured and stratified; and the permuted-entity null rules out the last structural explanation for the relational gain. The B1-card1 result stands as the project's one positive finding: $+0.00630$ PR-AUC at convergence, attributable to genuine entity history, with its direction supported by every ROC-AUC comparison and its exact magnitude bounded by a seed standard deviation the published bootstrap interval does not see.

Remaining work, in priority order:

1. **Per-feature ablation of the four relational summaries.** The gain is established as real but not localised: the four features are correlated $0.83$–$0.95$ and share $+0.00630$ between them. Runs at cap 15,000 with `average_precision` patience, scored against the converged references.
2. **Corrected cross-fitting control** with a shared encoder initialisation, repairing the defect recorded in Section 8. Independent of the B1 line.
3. **One-shot final test protocol**, once the ablation settles. The test partition has never been read and is evaluated exactly once, under a protocol written before it is opened.
4. **Combined multi-relation variant.** Low priority: card1 and card1_card2 are not distinguishable from each other on the paired bootstrap, so little is expected from combining them.
