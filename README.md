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
└── 6. Next Step: G1 Graph Neural Network (GNN) Neighborhood Embeddings
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

| Model Variant | Entity Definition | Predictors | Validation PR-AUC | Validation ROC-AUC | $\Delta \text{PR-AUC}_{\text{val}}$ vs B0 | Relational Feature Gain Ranks (/439) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`B0` (Baseline)** | Tabular Only | 435 | **0.64914** | **0.92502** | Baseline | — |
| **`B1-card_core_addr1`** | Composite Core + Addr1 | 439 | **0.64401** | **0.92348** | $-0.00513$ | #15, #18, #32, #49 |
| **`B1-card1`** | Card 1 Only | 439 | **0.65504** | **0.92807** | **+0.00590** | **#14, #16, #19, #31** |
| **`B1-card1_card2`** | Card 1 + Card 2 | 439 | **0.65465** | **0.92703** | **+0.00551** | **#15, #16, #18, #31** |

### Key Scientific Insights
1. **The Initial B1 Regression was Relational Selection Error**:
   - `card_core_addr1` fragmented the entity space (13.33% zero-fallback rate, 12.1h repeat gap), diluting temporal burst signals.
2. **Its Apparent Univariate Signal was Missingness, Not History**:
   - `card_core_addr1_prior_count` looks informative on the full training partition (ROC-AUC `0.3817` — a strong *inverse* association). Restricted to rows where the grouping key is actually present, its ROC-AUC is `0.5004`: no signal whatsoever.
   - The association is entirely the zero-count sentinel handed to uncovered rows, which carry a **10.18%** fraud rate against **2.49%** on covered rows. A bare "key is missing" flag scores PR-AUC `0.0609`, *above* the feature itself.
   - This is why `card_core_addr1` is shortlisted but never preferred: the screening requires a relation's history summaries to beat its own key-missingness indicator. See `reports/relational_screening/coverage_confound.csv`.
3. **High Coverage Restores Strong Positive Signal**:
   - Switching to `card1` (100% coverage) or `card1_card2` (98.42% coverage) yields a clean **+0.01103 PR-AUC recovery** over `card_core_addr1` and outperforms the tabular B0 baseline by **+0.00590 PR-AUC** (outcome `A` in `reports/b1/b1_cross_relation_summary.json`).
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
│   │   └── analyze_relations.py                       # Entity & graph diagnostic audit
│   ├── features/
│   │   ├── build_relational_features.py               # Generalized feature engineer
│   │   └── screen_relations.py                        # Train-only screening module
│   └── models/
│       ├── train_lightgbm_baseline.py                 # Frozen B0 LightGBM trainer
│       ├── train_lightgbm_relational.py               # Parametrized B1 LightGBM trainer
│       └── compare_b1_variants.py                     # Cross-variant comparison generator
└── tests/
    ├── test_lightgbm_baseline.py                      # B0 unit test suite
    ├── test_lightgbm_relational.py                    # B1 merge & invariant test suite
    ├── test_relational_features.py                    # Feature calculation unit tests
    ├── test_relational_screening.py                   # Stage A & B screening test suite (40 tests)
    └── test_relational_models.py                      # Multi-model verification suite (62 tests)
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

# 4. Generate cross-model comparison report
python -m src.models.compare_b1_variants

# 5. Run complete test suite (165 tests)
python -m pytest tests/ -v
```

---

## 8. Next Phase: Transition to G1 (Graph Neural Networks)

With `card1` established as the optimal relational entity substrate (100% coverage, 73.05% recurring entities, highest univariate & multivariate PR-AUC), the project proceeds to **G1**:
- **Graph Construction**: Transaction nodes connected via shared `card1` (and bipartite multi-relational edges).
- **GNN Neighborhood Aggregation**: Inductive message passing (GraphSAGE / Relational GCN) to capture multi-hop fraud rings that tree-based tabular models cannot represent.
