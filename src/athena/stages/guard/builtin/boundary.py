"""
stages.guard.builtin.boundary
==============================
BoundaryGuard — checks claims against declared boundaries.

A boundary is a negative prohibition: something the response was
explicitly forbidden to do, claim, or imply. A boundary violation is
categorically more serious than an unsubstantiated claim:

    unsubstantiated = the LLM made a claim it cannot support
    boundary violation = the LLM did something it was explicitly told not to

Boundary violations result in AdmissibilityState.inadmissible regardless
of risk level — they are contract breaches, not epistemic weaknesses.

The same design rationale as ConstraintGuard applies: natural language
boundaries require an LLM to evaluate reliably.

Key difference from ConstraintGuard
-------------------------------------
Boundaries apply to ALL claim types including normative claims.
"Do not recommend actions outside operator control" is a boundary that
applies specifically to normative claims. ConstraintGuard skips normative
claims; BoundaryGuard does not.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import GuardOutcome, GuardResult


# ---------------------------------------------------------------------------
# LLM result schema
# ---------------------------------------------------------------------------


class BoundaryCheckResult(BaseModel):
    """Schema for the boundary-checking LLM response."""

    violates: bool
    reasoning: str


_BOUNDARY_SYSTEM_PROMPT = """\
You are a boundary violation detector for an epistemic validation system.

You will be given:
- A boundary: a prohibition — something the response was forbidden to do,
  claim, or imply.
- A claim: a single atomic proposition extracted from the response.

Your task: determine whether this claim violates the boundary.

Rules:
- Return violates=true ONLY if the claim directly and clearly crosses
  the boundary. The violation must be evident from the claim text.
- Return violates=false if the claim does not cross the boundary,
  or if the boundary does not apply to this claim.
- Do not infer violations from implications — the violation must be
  present in the claim's proposition, not in what it might imply.
- Provide a brief one-sentence reasoning explaining your determination.
"""


# ---------------------------------------------------------------------------
# BoundaryGuard
# ---------------------------------------------------------------------------


class BoundaryGuard:
    """
    Checks each claim against all declared boundaries.

    Applies to all claim types including normative — boundaries
    often target prescriptive claims specifically.

    Parameters
    ----------
    model:
        PydanticAI Model instance for boundary classification.
    """

    def __init__(self, model: Model) -> None:
        self._agent = Agent[None, BoundaryCheckResult](
            model=model,
            output_type=BoundaryCheckResult,
            system_prompt=_BOUNDARY_SYSTEM_PROMPT,
            output_retries=2,
        )

    @property
    def guard_id(self) -> str:
        return "BoundaryGuard"

    async def evaluate(
        self,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> GuardResult:

        boundaries = decomposed_prompt.all_boundaries

        if not boundaries:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.skipped,
                message=(
                    "No boundaries declared (neither explicit nor implicit). "
                    "BoundaryGuard has nothing to check."
                ),
                detail={},
            )

        violated: list[str] = []
        reasonings: list[str] = []
        errors: list[str] = []

        for boundary in boundaries:
            user_message = (
                f"Boundary (prohibition): {boundary}\n\n"
                f"Claim proposition: {claim.proposition}\n"
                f"Claim type: {claim.claim_type}\n"
                f"Source span: {claim.source_span}"
            )
            try:
                result = await self._agent.run(user_message)
                check = result.output
                if check.violates:
                    violated.append(boundary)
                    reasonings.append(check.reasoning)
            except Exception as exc:  # noqa: BLE001 - any model/infra fault
                # An LLM outage is absence of evidence, not a violation.
                errors.append(f"{boundary} ({type(exc).__name__})")

        if violated:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    f"Claim violates {len(violated)} declared boundary/boundaries. "
                    f"First violation: '{violated[0]}'. "
                    f"Boundary violations result in inadmissibility."
                ),
                detail={
                    "violated_boundaries": "; ".join(violated),
                    "reasoning": "; ".join(reasonings),
                    "proposition": claim.proposition[:200],
                },
            )

        if errors:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.errored,
                message=(
                    f"Could not evaluate {len(errors)} of {len(boundaries)} "
                    f"declared boundary/boundaries (model error). The claim is "
                    f"unverified against them."
                ),
                detail={"unevaluated_boundaries": "; ".join(errors)},
            )

        return GuardResult(
            guard_id=self.guard_id,
            outcome=GuardOutcome.passed,
            message=(
                f"Claim does not violate any of the "
                f"{len(boundaries)} declared boundary/boundaries."
            ),
            detail={},
        )