"""
stages.assess.assessor
=======================
Step 4 — Admissibility Assessment.

Orchestrates all registered admissibility checkers against each claim
and synthesises their individual assessments into a single
AdmissibilityState + rationale per claim.

Checker execution order
-----------------------
1. StructuralAdmissibilityChecker (always first — reads guard results)
2. ModalityAdmissibilityChecker   (always second — risk-level adjustment)
3. Consumer-registered checkers   (in registration order)

The first checker to assign inadmissible terminates early — there is no
point running domain checkers on a structurally inadmissible claim.

Synthesis rules (precedence order)
------------------------------------
inadmissible  > contradicted > unsubstantiated > plausible > verified

The most severe state from any checker wins. A claim assessed as
`verified` by a domain checker but `contradicted` by another is
`contradicted` — the contradiction takes precedence.

One exception: `inadmissible` from StructuralAdmissibilityChecker
terminates assessment immediately. No subsequent checker can rescue
a structurally inadmissible claim.

Output
------
Produces a list of ClaimResult objects (one per claim) with:
- admissibility_state: synthesised from all checker results
- admissibility_rationale: combined narrative
- guard_results: carried through from the guard stage input
"""

from __future__ import annotations

import asyncio

from loguru import logger

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import (
    AdmissibilityState,
    ClaimResult,
    GuardResult,
)
from athena.core.protocols.admissibility import (
    AdmissibilityAssessment,
    AdmissibilityCheckerProtocol,
)
from athena.stages.assess.builtin.modality_admissibility import (
    ModalityAdmissibilityChecker,
)
from athena.stages.assess.builtin.structural_admissibility import (
    StructuralAdmissibilityChecker,
)

# Precedence order — higher index = more severe
_SEVERITY: dict[AdmissibilityState, int] = {
    AdmissibilityState.verified:        0,
    AdmissibilityState.plausible:       1,
    AdmissibilityState.unsubstantiated: 2,
    AdmissibilityState.contradicted:    3,
    AdmissibilityState.inadmissible:    4,
}


class AdmissibilityAssessor:
    """
    Orchestrates admissibility checkers for Step 4.

    Parameters
    ----------
    consumer_checkers:
        Domain-specific checkers registered by the consuming system.
        Run after built-in checkers.
    """

    def __init__(
        self,
        consumer_checkers: list[AdmissibilityCheckerProtocol] | None = None,
    ) -> None:
        self._structural = StructuralAdmissibilityChecker()
        self._modality   = ModalityAdmissibilityChecker()
        self._consumer_checkers: list[AdmissibilityCheckerProtocol] = (
            consumer_checkers or []
        )

    async def assess_all(
        self,
        claims: list[Claim],
        guard_results_map: dict[str, list[GuardResult]],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> list[ClaimResult]:
        """
        Assess all claims and produce ClaimResults.

        Parameters
        ----------
        claims:
            Extracted claims from Step 1.
        guard_results_map:
            claim_id → list[GuardResult] from Step 3 (GuardRunner).
        decomposed_prompt:
            Structured prompt decomposition.
        intent_context:
            Declared intent.

        Returns
        -------
        list[ClaimResult]
            One per claim, in the same order as input.
        """
        # Build partial ClaimResults with guard results only
        # (admissibility not yet assigned — assessor populates it)
        partial_results: list[ClaimResult] = [
            ClaimResult(
                claim=claim,
                guard_results=guard_results_map.get(claim.claim_id, []),
                admissibility_state=AdmissibilityState.plausible,
                admissibility_rationale="Pending assessment.",
            )
            for claim in claims
        ]

        final_results: list[ClaimResult] = []

        for i, claim in enumerate(claims):
            assessed = await self._assess_claim(
                claim=claim,
                partial_results=partial_results,
                decomposed_prompt=decomposed_prompt,
                intent_context=intent_context,
            )
            # Update partial_results so subsequent claims can see
            # assessments of earlier claims (for contradiction detection)
            partial_results[i] = assessed
            final_results.append(assessed)

        return final_results

    async def _assess_claim(
        self,
        claim: Claim,
        partial_results: list[ClaimResult],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> ClaimResult:
        """
        Run all checkers for a single claim and synthesise results.
        """
        guard_results = _find_guard_results(claim.claim_id, partial_results)
        assessments: list[AdmissibilityAssessment] = []

        # 1. Structural checker — always first
        structural_assessment = await _safe_assess(
            self._structural,
            claim, partial_results, decomposed_prompt, intent_context,
        )
        assessments.append(structural_assessment)

        # Early exit: structural inadmissibility cannot be rescued
        if structural_assessment.state == AdmissibilityState.inadmissible:
            return _build_claim_result(
                claim=claim,
                guard_results=guard_results,
                assessments=assessments,
                early_exit_reason="Structural guard failure — assessment terminated.",
            )

        # 2. Modality checker — always second
        modality_assessment = await _safe_assess(
            self._modality,
            claim, partial_results, decomposed_prompt, intent_context,
        )
        assessments.append(modality_assessment)

        # 3. Consumer checkers — concurrent for performance
        if self._consumer_checkers:
            consumer_tasks = [
                _safe_assess(checker, claim, partial_results,
                             decomposed_prompt, intent_context)
                for checker in self._consumer_checkers
            ]
            consumer_assessments = await asyncio.gather(*consumer_tasks)
            assessments.extend(consumer_assessments)

        return _build_claim_result(
            claim=claim,
            guard_results=guard_results,
            assessments=assessments,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _safe_assess(
    checker: AdmissibilityCheckerProtocol,
    claim: Claim,
    partial_results: list[ClaimResult],
    decomposed_prompt: DecomposedPrompt,
    intent_context: IntentContext,
) -> AdmissibilityAssessment:
    """
    Wrap checker.assess() in exception handling.
    A buggy checker cannot abort the pipeline.
    """
    try:
        assessment = await checker.assess(
            claim, partial_results, decomposed_prompt, intent_context
        )
        # Only built-in structural checkers may assign `inadmissible`; the
        # AdmissibilityAssessment constructor enforces it, this is the
        # belt-and-braces check for checkers that return a foreign object.
        if (
            assessment.state == AdmissibilityState.inadmissible
            and not getattr(assessment, "_internal", False)
        ):
            return AdmissibilityAssessment(
                checker_id=checker.checker_id,
                state=AdmissibilityState.contradicted,
                rationale=(
                    "Checker returned the reserved state 'inadmissible'; "
                    "demoted to contradicted. " + assessment.rationale
                ),
                evidence=dict(assessment.evidence),
            )
        return assessment
    except Exception as exc:
        logger.error(
            "Checker '{}' raised for claim '{}': {}",
            checker.checker_id, claim.claim_id, exc,
        )
        return AdmissibilityAssessment(
            checker_id=checker.checker_id,
            state=AdmissibilityState.unsubstantiated,
            rationale=(
                f"Checker raised an unexpected exception: {type(exc).__name__}. "
                f"Defaulting to unsubstantiated. See pipeline logs."
            ),
            evidence={"exception": str(exc)},
        )


def _build_claim_result(
    claim: Claim,
    guard_results: list[GuardResult],
    assessments: list[AdmissibilityAssessment],
    early_exit_reason: str | None = None,
) -> ClaimResult:
    """
    Synthesise assessments into a single ClaimResult.
    Most severe state wins. Rationale combines all assessor narratives.
    """
    if not assessments:
        return ClaimResult(
            claim=claim,
            guard_results=guard_results,
            admissibility_state=AdmissibilityState.unsubstantiated,
            admissibility_rationale="No assessments produced.",
        )

    # Pick the most severe state. `verified` (severity 0) can never win a
    # max(): it is positive evidence, so it is granted only when NO checker
    # returned anything worse than plausible and at least one returned verified.
    final_state = max(
        assessments,
        key=lambda a: _SEVERITY.get(a.state, 0),
    ).state
    if _SEVERITY[final_state] <= _SEVERITY[AdmissibilityState.plausible] and any(
        a.state == AdmissibilityState.verified for a in assessments
    ):
        final_state = AdmissibilityState.verified

    warnings: list[str] = []
    for a in assessments:
        warnings.extend(a.warnings)

    # Combine rationale from all checkers that contributed meaningfully
    rationale_parts: list[str] = []
    for assessment in assessments:
        if assessment.rationale:
            rationale_parts.append(
                f"[{assessment.checker_id}] {assessment.rationale}"
            )
    if early_exit_reason:
        rationale_parts.append(early_exit_reason)

    return ClaimResult(
        claim=claim,
        guard_results=guard_results,
        admissibility_state=final_state,
        admissibility_rationale=" | ".join(rationale_parts),
        warnings=warnings,
    )


def _find_guard_results(
    claim_id: str,
    partial_results: list[ClaimResult],
) -> list[GuardResult]:
    """Extract guard results for a claim_id from partial_results."""
    for cr in partial_results:
        if cr.claim.claim_id == claim_id:
            return list(cr.guard_results)
    return []