"""
stages.guard.builtin.constraint
================================
ConstraintGuard — checks claims against declared constraints.

A constraint is a positive obligation: something the response was
explicitly required to do or include. A claim that violates a constraint
is a contract breach — categorically different from being unsubstantiated.

Constraint checking is performed via an LLM call because constraints are
expressed in natural language and cannot be evaluated deterministically.

Key design: ConstraintGuard evaluates constraints at the RESPONSE level.
The full LLM response is passed in the user message so citation constraints
("must cite sensor ID for every numerical claim") can be evaluated correctly
even when the atomic claim proposition omits the citation. The citation may
exist in a neighbouring sentence of the same response.

Skips:
- Normative claims: constraints apply to epistemic content, not prescriptions.
- When no constraints are available.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from athena.core.models.claim import Claim, ClaimType
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import GuardOutcome, GuardResult


class ConstraintCheckResult(BaseModel):
    """Schema for the constraint-checking LLM response."""

    satisfies: bool
    reasoning: str


_CONSTRAINT_SYSTEM_PROMPT = """\
You are a constraint compliance checker for an epistemic validation system.

You will be given:
- A constraint: a positive obligation a response was required to satisfy.
- A claim proposition: an atomic statement extracted from the response.
- The full LLM response (when available): the complete text the claim came from.

Your task: determine whether this constraint is satisfied.

Rules:
- Evaluate at the RESPONSE level using the full response text when provided.
  Example: "Must cite sensor ID for numerical claims" — if the sensor ID
  appears ANYWHERE in the full response near the numerical value, the
  constraint is satisfied even if the atomic claim proposition omits it.
- Return satisfies=true if:
  (a) the constraint is met anywhere in the full response, or
  (b) this constraint does not apply to this type of claim at all.
- Return satisfies=false ONLY if the constraint is clearly violated even
  when considering the full response text.
- For citation, format, or timezone constraints: always check the full
  response text, not just the isolated claim proposition.
- Provide a brief one-sentence reasoning.
"""


class ConstraintGuard:
    """
    Checks each claim against all declared constraints.

    Evaluates constraints at the response level — the full LLM response
    is included in the check so citation and format constraints are not
    incorrectly flagged on atomic propositions that were correctly cited
    in a neighbouring sentence.
    """

    def __init__(self, model: Model) -> None:
        self._agent = Agent[None, ConstraintCheckResult](
            model=model,
            output_type=ConstraintCheckResult,
            system_prompt=_CONSTRAINT_SYSTEM_PROMPT,
            output_retries=2,
        )

    @property
    def guard_id(self) -> str:
        return "ConstraintGuard"

    async def evaluate(
        self,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> GuardResult:

        constraints = decomposed_prompt.all_constraints

        if not constraints:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.skipped,
                message="No constraints declared. ConstraintGuard has nothing to check.",
                detail={},
            )

        if claim.claim_type == ClaimType.normative:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.skipped,
                message=(
                    "Normative claim skipped: constraints apply to epistemic "
                    "content, not prescriptive recommendations."
                ),
                detail={"claim_type": claim.claim_type},
            )

        # Build full response context block for response-level constraint evaluation
        response_ctx = ""
        if decomposed_prompt.llm_response:
            response_ctx = (
                f"\n\nFull LLM response (evaluate citation/format/timezone "
                f"constraints against this complete text):\n---\n"
                f"{decomposed_prompt.llm_response[:2000]}\n---"
            )

        violated: list[str] = []
        reasonings: list[str] = []
        errors: list[str] = []

        for constraint in constraints:
            user_message = (
                f"Constraint: {constraint}\n\n"
                f"Claim proposition: {claim.proposition}\n"
                f"Claim type: {claim.claim_type}\n"
                f"Claim source span: {claim.source_span}"
                f"{response_ctx}"
            )
            try:
                result = await self._agent.run(user_message)
                check = result.output
                if not check.satisfies:
                    violated.append(constraint)
                    reasonings.append(check.reasoning)
            except Exception as exc:  # noqa: BLE001 - any model/infra fault
                # An LLM outage is ABSENCE of evidence, not evidence of a
                # violation. Record it as an error so it is neither reported
                # as a constraint breach nor silently treated as a pass.
                errors.append(f"{constraint} ({type(exc).__name__})")

        if violated:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    f"Claim violates {len(violated)} declared constraint(s). "
                    f"First violation: '{violated[0]}'"
                ),
                detail={
                    "violated_constraints": "; ".join(violated),
                    "reasoning": "; ".join(reasonings),
                    "proposition": claim.proposition[:200],
                },
            )

        if errors:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.errored,
                message=(
                    f"Could not evaluate {len(errors)} of {len(constraints)} "
                    f"declared constraint(s) (model error). The claim is "
                    f"unverified against them."
                ),
                detail={"unevaluated_constraints": "; ".join(errors)},
            )

        return GuardResult(
            guard_id=self.guard_id,
            outcome=GuardOutcome.passed,
            message=f"Claim satisfies all {len(constraints)} declared constraint(s).",
            detail={},
        )