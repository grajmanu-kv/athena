"""
stages.resolve.propagator
==========================
FailurePropagator — propagates admissibility failures through the
dependency graph produced by the builder.

Propagation rules
-----------------
The graph is traversed in topological order (dependencies before
dependents). For each claim:

depends_on propagation:
    If claim A depends on claim B, and B is inadmissible or contradicted,
    then A is downgraded:
        B = inadmissible  → A = inadmissible  (hard propagation)
        B = contradicted  → A = unsubstantiated (soft propagation —
                           A's evidence base is undermined but A itself
                           was not directly checked)
        B = unsubstantiated → no propagation (absence of evidence
                              about B does not transfer to A)

negates propagation:
    If claim A negates claim B (mutual contradiction), both are flagged.
    The claim with the weaker epistemic modality is downgraded:
        speculative/hypothetical < hedged < asserted
    If both are equally strong, the first in topological order is retained
    and the second is downgraded to contradicted.

propagated_failure flag:
    ClaimResults where admissibility was downgraded by propagation
    have propagated_failure=True. This distinguishes intrinsic failures
    (the claim itself failed a guard) from inherited failures (an upstream
    dependency failed).

Output
------
Returns a new list of ClaimResult objects with updated admissibility_state
and propagated_failure flags. Input ClaimResults are never mutated.
"""

from __future__ import annotations

from athena.core.models.claim import EpistemicModality
from athena.core.models.result import AdmissibilityState, ClaimResult
from athena.core.protocols.dependency import DependencyGraph

# Modality strength for negation resolution (lower index = weaker)
_MODALITY_STRENGTH: dict[EpistemicModality, int] = {
    EpistemicModality.speculative:  0,
    EpistemicModality.hypothetical: 1,
    EpistemicModality.hedged:       2,
    EpistemicModality.asserted:     3,
}

# States that trigger downstream propagation
_PROPAGATES_HARD: frozenset[AdmissibilityState] = frozenset({
    AdmissibilityState.inadmissible,
})
_PROPAGATES_SOFT: frozenset[AdmissibilityState] = frozenset({
    AdmissibilityState.contradicted,
})


class FailurePropagator:
    """
    Propagates admissibility failures through the dependency graph.
    Stateless — no constructor arguments.
    """

    def propagate(
        self,
        claim_results: list[ClaimResult],
        graph: DependencyGraph,
    ) -> list[ClaimResult]:
        """
        Apply failure propagation to all claims.

        Parameters
        ----------
        claim_results:
            Assessed ClaimResults from Step 4.
        graph:
            Dependency graph from the builder.

        Returns
        -------
        list[ClaimResult]
            New list with updated admissibility states.
            Input is not mutated.
        """
        # Build mutable state index: claim_id → ClaimResult
        result_map: dict[str, ClaimResult] = {
            cr.claim.claim_id: cr for cr in claim_results
        }

        # Get topological order — dependencies before dependents
        # A cyclic graph must NOT fall back to 'return unchanged': that would
        # silently skip failure propagation (fail-open). Let the error surface.
        order = graph.topological_order()

        # Process claims with no graph entries first (isolated claims)
        all_in_graph = graph.all_claim_ids()
        isolated_ids = [
            cr.claim.claim_id for cr in claim_results
            if cr.claim.claim_id not in all_in_graph
        ]

        # Process in topological order
        for claim_id in order:
            if claim_id not in result_map:
                continue
            result_map = _apply_depends_on_propagation(
                claim_id, result_map, graph
            )

        # Apply negation resolution
        result_map = _apply_negation_resolution(result_map, graph)

        # Reconstruct list in original order, recording the graph edges on the
        # (frozen) Claims via copies, as the Claim docstring promises.
        out: list[ClaimResult] = []
        for cr in claim_results:
            res = result_map.get(cr.claim.claim_id, cr)
            cid = res.claim.claim_id
            deps = sorted(graph.dependencies_of(cid))
            negs = sorted(graph.negations_of(cid))
            if deps != res.claim.depends_on or negs != res.claim.negates:
                res = res.model_copy(update={
                    "claim": res.claim.model_copy(
                        update={"depends_on": deps, "negates": negs}
                    )
                })
            out.append(res)
        return out


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------


def _apply_depends_on_propagation(
    claim_id: str,
    result_map: dict[str, ClaimResult],
    graph: DependencyGraph,
) -> dict[str, ClaimResult]:
    """
    For a claim, check all its dependencies and propagate failures.
    Returns updated result_map.
    """
    current = result_map.get(claim_id)
    if current is None:
        return result_map

    # Already inadmissible — no need to check dependencies
    if current.admissibility_state == AdmissibilityState.inadmissible:
        return result_map

    deps = graph.dependencies_of(claim_id)
    worst_propagated: AdmissibilityState | None = None
    propagation_reasons: list[str] = []

    for dep_id in deps:
        dep_result = result_map.get(dep_id)
        if dep_result is None:
            continue

        dep_state = dep_result.admissibility_state

        if dep_state in _PROPAGATES_HARD:
            worst_propagated = AdmissibilityState.inadmissible
            propagation_reasons.append(
                f"Upstream claim '{dep_id}' is inadmissible."
            )
            break  # hard propagation terminates early

        if dep_state in _PROPAGATES_SOFT:
            if worst_propagated != AdmissibilityState.inadmissible:
                worst_propagated = AdmissibilityState.unsubstantiated
                propagation_reasons.append(
                    f"Upstream claim '{dep_id}' is contradicted — "
                    f"this claim's evidence base is undermined."
                )

    if worst_propagated is None:
        return result_map

    # Downgrade only if propagated state is more severe
    current_severity = _severity(current.admissibility_state)
    propagated_severity = _severity(worst_propagated)

    if propagated_severity <= current_severity:
        return result_map

    updated = current.model_copy(
        update={
            "admissibility_state": worst_propagated,
            "admissibility_rationale": (
                current.admissibility_rationale
                + " | PROPAGATED: "
                + "; ".join(propagation_reasons)
            ),
            "propagated_failure": True,
        }
    )
    result_map = dict(result_map)
    result_map[claim_id] = updated
    return result_map


def _downgrade(result: ClaimResult, state: AdmissibilityState, note: str) -> ClaimResult:
    """Return `result` downgraded to `state`, but never to a LESS severe state."""
    if _severity(state) <= _severity(result.admissibility_state):
        return result
    return result.model_copy(
        update={
            "admissibility_state": state,
            "admissibility_rationale": result.admissibility_rationale + " | " + note,
            "propagated_failure": True,
        }
    )


def _apply_negation_resolution(
    result_map: dict[str, ClaimResult],
    graph: DependencyGraph,
) -> dict[str, ClaimResult]:
    """
    For each negation pair, downgrade the epistemically weaker claim to
    contradicted (both, on equal strength). A claim that is already more
    severe (inadmissible) keeps its state: negation never *upgrades*.
    Each pair is processed once.
    """
    processed_pairs: set[frozenset[str]] = set()
    result_map = dict(result_map)
    C = AdmissibilityState.contradicted

    for claim_id in list(result_map):
        for neg_id in graph.negations_of(claim_id):
            pair = frozenset({claim_id, neg_id})
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            a, b = result_map.get(claim_id), result_map.get(neg_id)
            if a is None or b is None:
                continue

            sa = _MODALITY_STRENGTH.get(a.claim.epistemic_modality, 0)
            sb = _MODALITY_STRENGTH.get(b.claim.epistemic_modality, 0)

            if sa < sb:
                result_map[claim_id] = _downgrade(
                    a, C,
                    f"NEGATION: Contradicted by claim '{neg_id}' which has "
                    f"stronger epistemic modality.",
                )
            elif sb < sa:
                result_map[neg_id] = _downgrade(
                    b, C,
                    f"NEGATION: Contradicted by claim '{claim_id}' which has "
                    f"stronger epistemic modality.",
                )
            else:
                note = "NEGATION: Mutually contradicted with a claim of equal modality strength."
                result_map[claim_id] = _downgrade(a, C, note)
                result_map[neg_id] = _downgrade(b, C, note)

    return result_map


def _severity(state: AdmissibilityState) -> int:
    _map = {
        AdmissibilityState.verified:        0,
        AdmissibilityState.plausible:       1,
        AdmissibilityState.unsubstantiated: 2,
        AdmissibilityState.contradicted:    3,
        AdmissibilityState.inadmissible:    4,
    }
    return _map.get(state, 0)