"""
stages.guard.builtin.terminology
==================================
TerminologyGuard — checks term usage consistency against declared definitions.

Terminological determinism: the prompt defines how key terms are used in
this context. Those definitions override general language and the LLM's
training distribution semantics. A claim that uses a defined term with a
different meaning than declared has drifted from the terminological frame
— this is a structural epistemic failure, not a factual one.

Two checks are performed:

1. Defined term usage consistency:
   For each term in DecomposedPrompt.term_definitions that appears in the
   claim's proposition, check whether the claim uses it consistently with
   its declared definition.

2. Undeclared term flagging:
   If a claim uses a term from DecomposedPrompt.undeclared_terms — terms
   flagged by the decomposition agent as potentially ambiguous — issue a
   warning. The ambiguity may or may not affect the claim's validity.

Both checks use an LLM for the same reason as constraint/boundary guards:
natural language definitions and propositions cannot be compared
deterministically with adequate precision.

Skips:
- Claims with no defined terms present in their proposition.
- When no term_definitions and no undeclared_terms are available.
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


class TerminologyCheckResult(BaseModel):
    """Schema for the terminology-checking LLM response."""

    consistent: bool
    reasoning: str


_TERMINOLOGY_SYSTEM_PROMPT = """\
You are a terminological consistency checker for an epistemic validation system.

You will be given:
- A term and its declared definition (the authoritative meaning in this context).
- A claim proposition that uses this term.

Your task: determine whether the claim uses the term consistently with
its declared definition.

Rules:
- Return consistent=true if the term is used in a way that is compatible
  with the declared definition, or if the term does not meaningfully appear
  in the claim.
- Return consistent=false if the claim clearly uses the term with a
  different meaning than declared — for example, using 'anomaly' in an
  everyday sense when it was declared to mean a specific statistical threshold.
- Minor phrasing differences are acceptable — focus on semantic meaning.
- Provide a brief one-sentence reasoning.
"""


# ---------------------------------------------------------------------------
# TerminologyGuard
# ---------------------------------------------------------------------------


class TerminologyGuard:
    """
    Checks term usage consistency against declared definitions.

    Parameters
    ----------
    model:
        PydanticAI Model instance for terminology classification.
    """

    def __init__(self, model: Model) -> None:
        self._agent = Agent[None, TerminologyCheckResult](
            model=model,
            output_type=TerminologyCheckResult,
            system_prompt=_TERMINOLOGY_SYSTEM_PROMPT,
            output_retries=2,
        )

    @property
    def guard_id(self) -> str:
        return "TerminologyGuard"

    async def evaluate(
        self,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> GuardResult:

        has_definitions = bool(decomposed_prompt.term_definitions)
        has_undeclared = bool(decomposed_prompt.undeclared_terms)

        if not has_definitions and not has_undeclared:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.skipped,
                message="No term definitions or undeclared terms. Nothing to check.",
                detail={},
            )

        inconsistent_terms: list[str] = []
        reasonings: list[str] = []
        undeclared_used: list[str] = []
        errors: list[str] = []

        # Check 1: defined term consistency
        if has_definitions:
            proposition_lower = claim.proposition.lower()
            for term, definition in decomposed_prompt.term_definitions.items():
                if term.lower() not in proposition_lower:
                    continue  # term not present in this claim

                user_message = (
                    f"Term: {term}\n"
                    f"Declared definition: {definition}\n\n"
                    f"Claim proposition: {claim.proposition}"
                )
                try:
                    result = await self._agent.run(user_message)
                    check = result.output
                    if not check.consistent:
                        inconsistent_terms.append(term)
                        reasonings.append(
                            f"'{term}': {check.reasoning}"
                        )
                except Exception as exc:  # noqa: BLE001 - any model/infra fault
                    # Absence of evidence, not evidence of drift.
                    errors.append(f"{term} ({type(exc).__name__})")

        # Check 2: undeclared term usage
        if has_undeclared:
            proposition_lower = claim.proposition.lower()
            for term in decomposed_prompt.undeclared_terms:
                if term.lower() in proposition_lower:
                    undeclared_used.append(term)

        # Build result
        if inconsistent_terms:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.failed,
                message=(
                    f"Claim uses {len(inconsistent_terms)} term(s) inconsistently "
                    f"with declared definitions: {', '.join(inconsistent_terms)}. "
                    f"This indicates semantic drift from the terminological frame."
                ),
                detail={
                    "inconsistent_terms": "; ".join(inconsistent_terms),
                    "reasoning": "; ".join(reasonings),
                    "proposition": claim.proposition[:200],
                },
            )

        if errors:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.errored,
                message=(
                    f"Could not check {len(errors)} defined term(s) "
                    f"(model error): {', '.join(errors)}. "
                    f"Terminology is unverified for this claim."
                ),
                detail={"unchecked_terms": "; ".join(errors)},
            )

        if undeclared_used:
            return GuardResult(
                guard_id=self.guard_id,
                outcome=GuardOutcome.warning,
                message=(
                    f"Claim uses {len(undeclared_used)} undeclared technical "
                    f"term(s) with no declared definition: "
                    f"{', '.join(undeclared_used)}. "
                    f"Meaning may be ambiguous — verify intended interpretation."
                ),
                detail={
                    "undeclared_terms_used": "; ".join(undeclared_used),
                    "proposition": claim.proposition[:200],
                },
            )

        return GuardResult(
            guard_id=self.guard_id,
            outcome=GuardOutcome.passed,
            message=(
                "All defined terms used consistently with declared definitions."
                + (
                    " No undeclared terms present in this claim."
                    if has_undeclared else ""
                )
            ),
            detail={},
        )