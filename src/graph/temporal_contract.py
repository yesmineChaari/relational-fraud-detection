"""Leakage-safe temporal message-passing contract for the G1 relational stage.

This module is the written specification the sampler (strictly-before temporal
neighbour sampler), the graph construction module, and the GraphSAGE encoder
are built and tested against. It produces no model code; every symbol below
either names a rule or operationalises one as a callable so it can be
asserted on directly, rather than re-derived by each downstream module.

1. Node and edge definition
----------------------------
A node is one transaction, identified by `TransactionID` and addressed
internally by a stable `node_id` (see `src/graph/build_transaction_graph.py`).
An edge exists between two transactions that share the same `card1` value —
the relation the project's own screening (Stage A/B) selected: 100% coverage,
73.05% recurring entities, the strongest covered-rows PR-AUC lift among the
audited candidates.

`card1` is a single, flat relation: every pair of transactions sharing a
`card1` value is equally "adjacent" — there is no second relation type a
2-hop walk could branch into. The full pairwise transaction-to-transaction
edge set for an entity is therefore a clique, and the largest entity is
around 14,500 transactions: a clique over it is on the order of 105 million
edges for that entity alone. Materialising that is the memory risk the
construction module exists to avoid (see its `STORAGE_RATIONALE`).

Consequently, this contract keeps structure and time separate:

* Structural adjacency (which transactions *could* ever be neighbours) is
  materialised once, as the entity-membership table the construction module
  writes — an O(N) bipartite structure (transaction -> its `card1` entity),
  not an O(entity_size^2) clique.
* Temporal admissibility (which of those transactions *is* a neighbour for a
  specific target, at a specific hop) is never materialised. It is induced
  by the sampler at sample time, because it is a function of the target
  being embedded, not of the source data alone. A fixed edge list computed
  once could not encode "admissible for this target" without being
  recomputed per target anyway.

2. Hop-1 admissibility
-----------------------
For a target transaction T with timestamp `t_T = TransactionDT(T)`, a
same-entity transaction N is an admissible hop-1 neighbour iff

    TransactionDT(N) < t_T                                   (strict)

This is exactly the "strictly-before" rule `compute_relational_features`
already enforces for the scalar relational features
(`src/features/build_relational_features.py`) and the temporal-edge audit
(`src/graph/analyze_relations.py`): a candidate at or after the target's own
timestamp is never observable to it.

3. Composition across hops
---------------------------
A hop-2 neighbour must be admissible with respect to the *target* T, not
merely with respect to the hop-1 node N it was reached through. Concretely,
for any node X reached by a path of any length rooted at T, admissibility is

    TransactionDT(X) < t_T                                   (strict)

evaluated against T's own timestamp at every hop, never against an
intermediate node's timestamp. `is_temporally_admissible` below takes the
target timestamp as an explicit argument for exactly this reason: it must be
threaded down through every hop of the walk unchanged, never replaced by the
current node's own timestamp as the recursion descends.

This is a deliberate choice between two bounds that are both leak-safe:

* **Target-anchored** (adopted): every node in T's computation graph, at any
  depth, satisfies `TransactionDT(node) < t_T`.
* **Intermediate-anchored** (rejected): a hop-2 node H must satisfy
  `TransactionDT(H) < TransactionDT(N)` for the specific hop-1 node N it was
  reached through.

Because `card1` is a single flat relation, every member of T's entity is
already equally "adjacent" to T — hop-1 and hop-2 both draw from the same
entity, just reached by a different sampling path. Intermediate-anchoring
would exclude a same-entity transaction H with `t_N < t_H < t_T` purely
because the sampler happened to reach it through N instead of directly,
even though H is exactly as legitimate a piece of T's history as N is. That
exclusion buys no additional leakage safety — target-anchoring already
guarantees `t_H < t_T` — it only produces an arbitrary, sampling-order-
dependent neighbourhood. Target-anchoring is therefore both sufficient for
safety and the non-arbitrary choice; it is also what "strictly-before T"
already means everywhere else in this codebase.

4. Equal-timestamp handling
----------------------------
Ties are excluded, matching the strictly-before block semantics already in
place: a same-entity transaction at exactly `t_T` never observes, and is
never observed by, another transaction in its own timestamp block. `<` is
never relaxed to `<=` at any hop.

5. Transductive vs. inductive training
----------------------------------------
The encoder is trained inductively (`TRAINING_REGIME`): it computes a node's
embedding purely from its own features and its sampled neighbours' features,
never from a learned per-node embedding table. This is required by G1's own
scope (FRM-10: an inductive GraphSAGE encoder) and by the dataset's
mechanics — validation and test nodes must be embeddable without having been
seen at training time.

Because entity history is continuous across splits (never reset at split
boundaries, the same policy the scalar relational features already use), a
validation or test target's admissible neighbourhood routinely extends back
into the train period, and a test target's neighbourhood may include
validation-period transactions. This is expected and required, not a leak:
admissibility is still governed strictly by rule 3 above, evaluated against
the target's own timestamp, regardless of which split a neighbour's
timestamp happens to fall in.

Training itself only ever backpropagates through train-partition target
nodes, mirroring B0/B1's discipline of fitting on train and evaluating on
validation only. Because a train target's admissible neighbours must be
strictly before its own timestamp, and train is the earliest partition
chronologically, no validation or test information can reach a training
gradient through this route either.

6. Node feature vector contents
----------------------------------
A node's feature vector may contain the same predictors the frozen B0
manifest already licenses (`reports/baseline/baseline_metadata.json` ->
`feature_columns`), encoded with the same frozen, train-fitted categorical
mappings B1 already reuses unchanged
(`reports/baseline/categorical_mappings.json`). `FORBIDDEN_NODE_FEATURE_COLUMNS`
below is the graph-domain restatement of B0's `FORBIDDEN_FEATURE_COLUMNS`.

`isFraud` is singled out and must never be reachable, at any hop, for any
node's feature vector — including a temporally admissible neighbour's. This
is a distinct rule from temporal admissibility: a same-entity neighbour can
be strictly before the target in time and still leak its label into the
target's computation graph, because relational fraud-ring signal is exactly
a correlation between neighbours' labels. Temporal admissibility governs
whether a node may be looked at; this rule governs what may be looked at on
it. `assert_no_forbidden_node_features` enforces both at once.
"""

from __future__ import annotations

from typing import Iterable

RELATION_NAME = "card1"
GROUP_COLUMNS = ["card1"]

# Restates train_lightgbm_baseline.FORBIDDEN_FEATURE_COLUMNS for the graph
# domain so this module has no import-time dependency on the model trainers.
FORBIDDEN_NODE_FEATURE_COLUMNS = frozenset(
    {
        "TransactionID",
        "isFraud",
        "split",
        "has_identity",
    }
)

TRAINING_REGIME = "inductive"

EDGE_STRUCTURAL_STORAGE = "entity_to_transaction_bipartite"
EDGE_TEMPORAL_ADMISSIBILITY = "induced_by_sampler"


def is_temporally_admissible(node_dt: float, target_dt: float) -> bool:
    """Rule 2/3: strictly-before, evaluated against the target at every hop.

    `target_dt` must always be the timestamp of the node whose embedding is
    ultimately being computed, never an intermediate hop's own timestamp.
    """
    return node_dt < target_dt


def validate_sampled_neighborhood(
    target_dt: float,
    neighbor_dts: Iterable[float],
) -> None:
    """Raise if any sampled neighbour, at any hop, violates rule 3."""
    violations = [dt for dt in neighbor_dts if not is_temporally_admissible(dt, target_dt)]
    if violations:
        raise AssertionError(
            f"{len(violations)} sampled neighbour(s) are not strictly before "
            f"the target timestamp {target_dt!r}: {sorted(violations)[:10]}."
        )


def assert_no_forbidden_node_features(feature_columns: Iterable[str]) -> None:
    """Rule 6: isFraud and the other B0-forbidden columns may never appear."""
    leaked = FORBIDDEN_NODE_FEATURE_COLUMNS & set(feature_columns)
    if leaked:
        raise AssertionError(f"Forbidden columns present in node feature vector: {sorted(leaked)}.")
