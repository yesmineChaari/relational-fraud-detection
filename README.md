# Relational Fraud Detection

Does a transaction's **entity history** (the earlier transactions that share its card, address or device) improve a strong tabular fraud model? This repository tests that question on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset (590,540 transactions). It compares a LightGBM baseline, scalar relational features and inductive graph neural networks, and fixes every verdict rule before its run so that validation never becomes a selection loop.

**Short answer:** yes, a little, and only as simple features. Four history summaries over `card1` add **+0.0063 PR-AUC** on validation and **+0.0043** on the held-out test split. None of the graph approaches tried beat those four features: two GraphSAGE encoders and a finer user-ID key.

## Contents

- [Results](#results)
- [Key findings](#key-findings)
- [Methodology](#methodology)
- [Getting started](#getting-started)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)

## Results

Validation PR-AUC unless marked (fraud rate 3.4%). Every delta is a paired bootstrap over validation rows with 10,000 resamples.

| Experiment | Question | Reference | Δ PR-AUC | 95% CI | Outcome |
| :--- | :--- | :--- | ---: | :--- | :--- |
| **B1-card1** | Do four `card1` history summaries help? | B0 | **+0.00630** | [+0.00239, +0.01018] | **Gain** |
| B1-card1, test split | Does the gain hold on unseen data? | B0 | +0.00425 | [+0.00020, +0.00837] | Replicates, narrowly |
| B1-card_core_addr1 | Does a more specific card key help? | B0 | −0.00513 | — | Worse |
| G1 | Does a GraphSAGE embedding over `card1` help? | B0 | −0.02093 | [−0.02632, −0.01540] | Worse |
| G1-v2 | Does a count-aware, cross-fitted encoder close the gap? | B1-card1 ¹ | −0.02880 (3-seed mean) | excludes 0 on every seed | Worse |
| + `addr1` | Does a second relation add anything? | B1-card1 ¹ | −0.00263 | [−0.00610, +0.00079] | No gain |
| + `device_fingerprint` | Does a device relation add anything? | B1-card1 ¹ | −0.00049 | [−0.00314, +0.00209] | No gain |
| + `uid` (G2) | Does a finer user-ID key add anything? | B1-card1 ¹ | −0.00114 | [−0.00389, +0.00161] | No gain; graph not built |

¹ B1-card1 retrained for exactly 10,000 rounds without early stopping (PR-AUC 0.66138), so both arms get the same budget.

At convergence, B0 scores 0.64919 and B1-card1 scores 0.65550. On the test split both drop by about 0.09 PR-AUC, so the test figures measure the same effect at a lower level.

## Key findings

1. **The gain is genuine entity history.** Permuting `card1` across transactions keeps every count and timestamp distribution but breaks the link between a transaction and its entity, and that removes the gain: no permuted run beats B0 ([`reports/permuted_null/`](reports/permuted_null/)).
2. **Coverage beats specificity.** `card_core_addr1` lost to B0 because 13% of rows have no key and get a zero-count sentinel. Those rows are four times as likely to be fraud, so the feature learned missingness, not history. Screening now requires a relation's history signal to beat a bare "key is missing" flag.
3. **Graph embeddings lost to four counts.** Most of G1's deficit was width: a shuffled 32-column block alone costs −0.0196, 86% of the gap. After alignment, G1-v2's cross-fitted embedding scored the same as a random block of the same width.
4. **A strong neighbour-label lift does not guarantee useful features.** Neighbours under the `uid` key (`card1 | addr1 | account start day`) share the target's label 99% of the time: a 24× fraud lift, against 8.5× for `card1`. Yet its label-free history added nothing beyond `card1`.
5. **The count summaries are additive.** At a fixed budget, `prior_count` and `prior_count_7d` each contribute independently. Recency adds nothing on its own ([`reports/fixed_budget/`](reports/fixed_budget/)).
6. **Measurement artifacts were larger than the effect.** Before correction:
   - the seed-to-seed spread of the gain was three times the bootstrap interval;
   - the 6,000-round cap was binding;
   - the reported PR-AUC was a validation argmax.

   Each artifact was measured and corrected ([`seed_variance/`](reports/seed_variance/), [`convergence_check/`](reports/convergence_check/), [`selection_bias/`](reports/selection_bias/)).
7. **The gain sits at one operating point.** Most of it lands near 400 alerts per day: 46 extra frauds caught over 31 days, and none at 25 alerts per day. Despite class weighting, scores under-predict the base rate: the mean score is 0.64× prevalence ([`reports/operating_points/`](reports/operating_points/)).

## Methodology

| Principle | How it is enforced |
| :--- | :--- |
| Temporal split | 70/15/15 by `TransactionDT`: 413,378 train, 88,581 validation, 88,581 test rows |
| One change at a time | Every variant reuses B0's frozen LightGBM parameters and train-only categorical mappings. Only the feature manifest changes, and reference artifacts are hash-pinned before and after each run |
| No leakage | Features see only strictly earlier transactions (equal timestamps never see each other) and never read `isFraud`. The graph sampler enforces the same strictly-before rule at every hop ([`src/graph/temporal_contract.py`](src/graph/temporal_contract.py)) |
| Paired statistics | One shared paired bootstrap ([`src/models/significance.py`](src/models/significance.py)) runs on persisted validation predictions, so an interval never needs a refit |
| Pre-registration | Each verdict rule is committed before its run ([`configs/`](configs/)). A failed gate stops the plan, and seeds are never re-drawn |
| Fixed budgets | Later comparisons train exactly 10,000 rounds without early stopping, which removes argmax selection bias |
| One test read | The test split was scored once, under a committed protocol ([`src/models/final_test_protocol.py`](src/models/final_test_protocol.py)). The executor now refuses to run again |

The frozen LightGBM configuration is: learning rate 0.03, 64 leaves, `min_child_samples` 50, 0.8 column and row subsampling, `reg_alpha` 0.1, `reg_lambda` 1.0, and `scale_pos_weight` 27.4. The authoritative values are in [`reports/baseline/`](reports/baseline/).

## Getting started

**Requirements:** Python 3.13, [Git LFS](https://git-lfs.com/) for the model files and prediction files, and about 16 GB of RAM (each training run peaks near 4.5 GB).

```bash
git clone https://github.com/yesmineChaari/relational-fraud-detection.git
cd relational-fraud-detection
git lfs pull

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

**Data.** The raw data (about 1.3 GB) is not redistributed. Accept the competition rules on [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place `train_transaction.csv` and `train_identity.csv` in `data/raw/`, then verify them:

```bash
python -m src.data.verify_raw_inputs
```

The competition's `test_*.csv` files carry no labels and are not used. This project's test split is carved from the labelled training file.

## Reproducing the experiments

The core pipeline runs from raw inputs to the B1 comparison. It checks prerequisites up front and skips stages whose outputs already exist:

```bash
python -m src.pipeline --all             # verify → profile → split → dataset → audit → screening → features → B0/B1
python -m src.pipeline --list            # show the stages
python -m src.pipeline --all --dry-run   # show what would run
python -m src.pipeline --from relational_features
```

The investigation stages are separate enquiries and run directly. Training runners accept `--skip-existing`, so an interrupted panel resumes where it stopped. Run one training process at a time.

<details>
<summary>All investigation commands</summary>

```bash
# G1: card1 graph, GraphSAGE encoder, LightGBM on the embedding, attribution controls
python -m src.graph.build_transaction_graph
python -m src.graph.train_graphsage_encoder
python -m src.models.train_lightgbm_g1
python -m src.graph.train_graphsage_variants --skip-existing
python -m src.models.train_lightgbm_g1_controls --skip-existing
python -m src.models.compare_g1_controls

# Seed variance, convergence at a 15,000-round cap, permuted-entity null
python -m src.models.train_seed_variants --skip-existing
python -m src.models.summarize_seed_variance
python -m src.models.train_lightgbm_convergence_check --skip-existing
python -m src.models.summarize_convergence_check
python -m src.models.train_lightgbm_permuted_null --skip-existing
python -m src.models.compare_permuted_null

# Per-feature ablation, selection-bias screen, fixed-budget refits
python -m src.models.train_lightgbm_ablation --skip-existing
python -m src.models.compare_ablation
python -m src.models.compare_selection_bias
python -m src.models.rederive_ablation_at_equal_budget
python -m src.models.train_ablation_fixed_budget --skip-existing
python -m src.models.compare_fixed_budget_ablation

# Cross-stage table, operating points, experiment ledger (no refits)
python -m src.models.compare_converged_significance
python -m src.models.compare_stages
python -m src.models.calibration_and_operating_points
python -m src.models.build_experiment_ledger

# Extra relations on top of B1-card1 (addr1, device_fingerprint, uid)
python -m src.graph.analyze_relations
python -m src.features.screen_relations
python -m src.features.build_relational_features --relation uid
python -m src.models.train_lightgbm_stage0_check --relation uid --skip-existing
python -m src.models.compare_stage0_candidates

# G1-v2: pre-registered seed panel, then the post-hoc alignment check
python -m src.models.train_g1v2_seed_panel
python -m src.models.diagnose_g1v2_alignment --step build
python -m src.models.diagnose_g1v2_alignment --step evaluate
```

</details>

### Tests

```bash
python -m pytest -m "not benchmark" -q
```

The suite has 927 tests: 925 in the gating run and two wall-clock benchmarks that are excluded from it. Continuous integration runs `ruff check`, `ruff format --check` and the gating suite on every push. The 53 tests that need the raw data skip on a clean checkout.

## Repository layout

```
├── configs/          Screening thresholds and pre-registered designs
├── data/             raw/ (not tracked) and processed/ (generated)
├── models/           Every trained model (Git LFS), indexed by the experiment ledger
├── reports/          Metrics, predictions, comparisons and verdicts, one folder per enquiry
├── src/
│   ├── data/         Raw-input checks, temporal split, model dataset
│   ├── features/     Relational features, screening, the shared uid key definition
│   ├── graph/        Relation audit, temporal sampler, graph builders, GraphSAGE encoders
│   ├── models/       Trainers, controls, significance, verdicts, ledger
│   └── pipeline.py   Orchestrated core pipeline
└── tests/            Unit and integration tests
```

[`reports/ledger/`](reports/ledger/) indexes all 83 models with their stage, purpose, the claims resting on them, and what they must not be used for.

## Limitations

- **Research models, not a deployable system.** The data is a 2019 competition set, models are selected on validation, and scores are not probabilities.
- **The effect is small.** The +0.0063 gain is close to the refit noise floor, so its magnitude is not quotable to three decimals from one run.
- **The test split is spent.** It was read once, for B0 and B1-card1. Every other number is a validation result, and a further model change needs a newly held-out partition.
- **The frozen artifacts sit at a 6,000-round cap** so that published numbers stay reproducible. Converged figures supersede them where they differ.
- **Cost analysis is per incident.** Transaction amounts are not weighted.
