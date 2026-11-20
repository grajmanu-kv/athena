"""
stages.guard.builtin.structural
================================
StructuralGuard — validates claim schema completeness per ClaimType.

This is the hardest of the hard invariants. It enforces the Popperian
falsifiability requirement at the claim level:

    - A predictive claim without a time_horizon is unfalsifiable.
    - A causal claim without assumptions has no mechanism — unfalsifiable.
    - A claim with no scope cannot be verified against domain knowledge.

These checks are also performed by Pydantic model validators, but
StructuralGuard re-enforces them as an explicit pipeline stage so they:
    1. Appear in GuardResult and therefore in the audit trail.
    2. Can be inspected per-claim by consuming systems.
    3. Are not silently swallowed if a Claim was constructed via
       model_construct() in tests or adversarial contexts.

A failed StructuralGuard result will cause AdmissibilityAssessor to
assign AdmissibilityState.inadmissible to the claim — it cannot be
rescued by domain checkers or dependency graph propagation.
"""

from __future__ import annotations

from athena.core.models.claim import Claim, ClaimType
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import GuardOutcome, GuardResult
from athena.util.duration import parse_duration_seconds


class StructuralGuard:
    """
    Validates claim schema completeness per ClaimType.

    Hard invariants enforced:
    - predictive claims must have time_horizon
    - causal claims must have at least one assumption
    - all claims must have a non-empty, non-trivial scope
    - all claims must have a non-empty source_span
    - no compound propositions (detected heuristically)

    Soft warnings:
    - predictive claims with unquantified time_horizon
    - predictive claims whose quantified time_horizon exceeds a declared,
      parseable decision window (the prediction lands after the decision)
    - proposition that looks compound
    """

    @property
    def guard_id(self) -> str:
        return "StructuralGuard"

    async def evaluate(
        self,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> GuardResult:

        # Hard: source_span must be non-trivially present
        if not claim.source_span or len(claim.source_span.strip()) < 3:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    "Claim has no traceable source_span. "
                    "Every claim must be anchored to a verbatim fragment "
                    "of the LLM response. This claim is epistemically unanchored."
                ),
                detail={"claim_id": claim.claim_id},
            )

        # Hard: predictive requires time_horizon
        if claim.claim_type == ClaimType.predictive and claim.time_horizon is None:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    "Predictive claim has no time_horizon. "
                    "A prediction without a time horizon cannot be falsified "
                    "and is therefore epistemically inadmissible."
                ),
                detail={
                    "claim_type": claim.claim_type,
                    "proposition": claim.proposition[:120],
                },
            )

        # Hard: causal requires assumptions
        if claim.claim_type == ClaimType.causal and not claim.assumptions:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    "Causal claim has no assumptions. "
                    "A causal claim without a declared mechanism is unfalsifiable. "
                    "At least one assumption documenting the causal mechanism "
                    "is required."
                ),
                detail={
                    "claim_type": claim.claim_type,
                    "proposition": claim.proposition[:120],
                },
            )

        # Hard: scope must not be empty
        if not claim.scope or len(claim.scope.strip()) < 3:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    "Claim has no scope. "
                    "Every claim must declare its domain, temporal, "
                    "and spatial context."
                ),
                detail={"claim_id": claim.claim_id},
            )

        # Soft: unquantified time_horizon on predictive claim
        if (
            claim.claim_type == ClaimType.predictive
            and claim.time_horizon is not None
            and not claim.time_horizon.is_quantified
        ):
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.warning,
                message=(
                    f"Predictive claim has a vague time_horizon: "
                    f"'{claim.time_horizon.raw}'. "
                    f"Could not normalise to ISO 8601. "
                    f"The claim is structurally admissible but epistemically weak — "
                    f"it cannot be verified against the decision window."
                ),
                detail={
                    "time_horizon_raw": claim.time_horizon.raw,
                    "normalized": str(claim.time_horizon.normalized),
                },
            )

        # Soft: horizon longer than the decision window
        if (
            claim.claim_type == ClaimType.predictive
            and claim.time_horizon is not None
            and claim.time_horizon.is_quantified
        ):
            horizon_s = parse_duration_seconds(claim.time_horizon.normalized)
            window_s = parse_duration_seconds(intent_context.decision_window)
            if horizon_s is not None and window_s is not None and horizon_s > window_s:
                return GuardResult(
                    guard_id=self.guard_id,
                    outcome=GuardOutcome.warning,
                    message=(
                        f"Predictive time_horizon '{claim.time_horizon.raw}' "
                        f"({claim.time_horizon.normalized}) is longer than the "
                        f"declared decision window "
                        f"'{intent_context.decision_window}'. The prediction "
                        f"resolves after the decision must be made."
                    ),
                    detail={
                        "time_horizon_raw": claim.time_horizon.raw,
                        "normalized": str(claim.time_horizon.normalized),
                        "decision_window": str(intent_context.decision_window),
                    },
                )

        # Soft: compound proposition heuristic
        # A proposition containing " and " linking two independent clauses
        # suggests the claim was not properly atomised.
        if _looks_compound(claim.proposition):
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.warning,
                message=(
                    "Proposition may be compound — it appears to contain "
                    "more than one independent assertion. "
                    "Compound claims cannot be independently evaluated. "
                    "Consider splitting at extraction time."
                ),
                detail={"proposition": claim.proposition[:200]},
            )

        return GuardResult(
            guard_id=self.guard_id,
            outcome=GuardOutcome.passed,
            message=(
                f"Structural check passed: {claim.claim_type} claim "
                f"has all required fields for its type."
            ),
            detail={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_compound(proposition: str) -> bool:
    """
    Heuristic: does the proposition contain two independent clauses?

    Checks for conjunctions that commonly link independent propositions.
    Not a hard failure — a warning only. Natural language is ambiguous
    and this check will have false positives.
    """
    lower = proposition.lower()
    # Patterns that suggest two independent propositions joined together
    compound_patterns = [
        " and therefore ",
        " and as a result ",
        " and consequently ",
        " which means that ",
        " which caused ",
        " thus ",
        " hence ",
    ]
    return any(pattern in lower for pattern in compound_patterns)