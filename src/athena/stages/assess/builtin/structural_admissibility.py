"""
stages.assess.builtin.structural_admissibility
===============================================
StructuralAdmissibilityChecker — derives AdmissibilityState from
the structural guard results already attached to each ClaimResult.

This checker does not call an LLM and does not consult domain knowledge.
It reads the GuardResults produced by the guard stage and translates them
into an initial admissibility state following a strict precedence order:

    Any GuardOutcome.failed from StructuralGuard
        → AdmissibilityState.inadmissible

    Any GuardOutcome.failed from BoundaryGuard
        → AdmissibilityState.inadmissible

    Any GuardOutcome.failed from ConstraintGuard
        → AdmissibilityState.contradicted
        (constraint violation is a contract breach — stronger than
        unsubstantiated but the claim may still be factually true)

    Any GuardOutcome.failed from TerminologyGuard
        → AdmissibilityState.unsubstantiated
        (semantic drift weakens the claim's epistemic grounding)

    Any GuardOutcome.failed from an unknown (consumer) guard
        → AdmissibilityState.contradicted
        (a failed guard must never be silently ignored)

    Several failures: the MOST SEVERE state wins, independent of guard order.

    Any GuardOutcome.errored (guard could not run, e.g. LLM outage)
        → AdmissibilityState.unsubstantiated
        (absence of evidence: neither a violation nor a pass)

    Any GuardOutcome.warning (any guard)
        → AdmissibilityState.plausible
        (soft issues — claim can proceed with caveats)

    No failures or warnings
        → AdmissibilityState.plausible
        (structural baseline — domain checkers may upgrade to verified)

Rationale for not assigning `verified` here
--------------------------------------------
StructuralAdmissibilityChecker cannot verify claims — it can only detect
structural failures. `verified` requires positive evidence from domain
knowledge (known_facts, injected_facts). That is the responsibility of
domain-specific AdmissibilityCheckerProtocol implementations registered
by consuming systems.

The AdmissibilityAssessor (assessor.py) synthesises results from this
checker and all registered domain checkers to produce the final state.
"""

from __future__ import annotations

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import (
    AdmissibilityState,
    ClaimResult,
    GuardOutcome,
)
from athena.core.protocols.admissibility import _BUILTIN, AdmissibilityAssessment

# Guard IDs that cause inadmissibility on failure
_INADMISSIBLE_ON_FAILURE: frozenset[str] = frozenset({
    "StructuralGuard",
    "BoundaryGuard",
})

# Guard IDs that cause contradicted on failure
_CONTRADICTED_ON_FAILURE: frozenset[str] = frozenset({
    "ConstraintGuard",
})

# Guard IDs that cause unsubstantiated on failure
_UNSUBSTANTIATED_ON_FAILURE: frozenset[str] = frozenset({
    "TerminologyGuard",
})


class StructuralAdmissibilityChecker:
    """
    Derives initial AdmissibilityState from guard results.

    Stateless — no constructor arguments.
    Must run before domain admissibility checkers so they have a
    baseline state to work from.
    """

    @property
    def checker_id(self) -> str:
        return "StructuralAdmissibilityChecker"

    async def assess(
        self,
        claim: Claim,
        all_claim_results: list[ClaimResult],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> AdmissibilityAssessment:
        """
        Derive admissibility from guard results already attached to
        the ClaimResult for this claim.
        """
        # Find the ClaimResult for this claim
        claim_result = _find_claim_result(claim.claim_id, all_claim_results)

        if claim_result is None:
            # No guard results yet — this should not happen in normal
            # pipeline flow, but handle defensively.
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=AdmissibilityState.unsubstantiated,
                rationale=(
                    "No guard results found for this claim. "
                    "Cannot assess structural admissibility."
                ),
            )

        guard_results = claim_result.guard_results

        # Collect EVERY hard failure and take the most severe. (Returning on
        # the first failure in guard order made the outcome depend on guard
        # registration order: a TerminologyGuard failure listed before a
        # StructuralGuard failure would mask the inadmissible state.)
        failures = [g for g in guard_results if g.outcome == GuardOutcome.failed]
        candidates: list[tuple[int, AdmissibilityState, GuardResult, str]] = []
        for gr in failures:
            if gr.guard_id in _INADMISSIBLE_ON_FAILURE:
                candidates.append((4, AdmissibilityState.inadmissible, gr, ""))
            elif gr.guard_id in _CONTRADICTED_ON_FAILURE:
                candidates.append((
                    3, AdmissibilityState.contradicted, gr,
                    " Constraint violation is a contract breach.",
                ))
            elif gr.guard_id in _UNSUBSTANTIATED_ON_FAILURE:
                candidates.append((
                    2, AdmissibilityState.unsubstantiated, gr,
                    " Terminological drift weakens epistemic grounding.",
                ))
            else:
                # Consumer guard: its failure must not be silently ignored.
                candidates.append((
                    3, AdmissibilityState.contradicted, gr,
                    " Failed a registered consumer guard.",
                ))

        if candidates:
            _, state, gr, suffix = max(candidates, key=lambda c: c[0])
            others = [c[2].guard_id for c in candidates if c[2] is not gr]
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=state,
                rationale=f"{gr.guard_id} failed: {gr.message}{suffix}"
                + (f" (also failed: {', '.join(others)})" if others else ""),
                evidence={
                    "guard_id": gr.guard_id,
                    "guard_message": gr.message,
                },
                _internal=_BUILTIN if state == AdmissibilityState.inadmissible else None,
            )

        # A guard that could not run is absence of evidence: the claim is not
        # shown to be bad, but it is not shown to be good either.
        errored = [g for g in guard_results if g.outcome == GuardOutcome.errored]
        if errored:
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=AdmissibilityState.unsubstantiated,
                rationale=(
                    "Could not complete verification: "
                    + "; ".join(f"{g.guard_id} errored ({g.message})" for g in errored[:2])
                    + ("..." if len(errored) > 2 else "")
                ),
                evidence={"errored_guards": ", ".join(g.guard_id for g in errored)},
            )

        # Check for soft warnings — downgrade to plausible with caveat
        warning_messages = [
            gr.message for gr in guard_results
            if gr.outcome == GuardOutcome.warning
        ]
        if warning_messages:
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=AdmissibilityState.plausible,
                rationale=(
                    "Claim passed all hard guard checks but has soft warnings: "
                    + "; ".join(warning_messages[:2])
                    + ("..." if len(warning_messages) > 2 else "")
                ),
                evidence={"warning_count": str(len(warning_messages))},
            )

        # No failures, no warnings — structural baseline is plausible
        # Domain checkers may upgrade to verified
        return AdmissibilityAssessment(
            checker_id=self.checker_id,
            state=AdmissibilityState.plausible,
            rationale=(
                "Claim passed all structural guard checks with no failures "
                "or warnings. Baseline admissibility: plausible. "
                "Domain checkers may upgrade to verified."
            ),
        )


def _find_claim_result(
    claim_id: str,
    claim_results: list[ClaimResult],
) -> ClaimResult | None:
    """Find the ClaimResult for a given claim_id."""
    for cr in claim_results:
        if cr.claim.claim_id == claim_id:
            return cr
    return None