"""
stages.extract.agent
====================
Step 1 — Claim Extraction.

Uses util.Prompt to load system.md (static, cached) and render
message.md (per-call, with $variable substitution) for the user turn.

PydanticAI API (pydantic-ai >= 0.0.20)
---------------------------------------
- output_type=list[Claim]  (direct list, no wrapper model)
- result.output            (not result.data)
- output_retries           controls schema-validation retry count
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import Model

from athena.core.exceptions import ClaimExtractionFailure
from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.util.prompt import Prompt, fence_for

_PROMPT = Prompt("extract")


# ---------------------------------------------------------------------------
# Extraction warning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionWarning:
    """
    Non-fatal issue encountered during claim extraction.
    Surfaces in AuditReport.pipeline_warnings.
    """

    reason: str
    source_fragment: str | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ExtractAgent:
    """
    Step 1 — extracts atomic epistemic claims from an LLM response.

    Parameters
    ----------
    model:
        A pydantic_ai Model instance. Provided by the calling system.
    output_retries:
        Retry count on schema validation failure. Default: 3.
    verify_source_spans:
        When True (default), excludes claims whose source_span does not
        appear verbatim in llm_response.
    """

    def __init__(
        self,
        model: Model,
        output_retries: int = 3,
        verify_source_spans: bool = True,
    ) -> None:
        self._output_retries = output_retries
        self._verify_source_spans = verify_source_spans
        # Explicit generic parameters force the type checker to resolve
        # Agent[None, list[Claim]] correctly.
        # Without this, pyright/mypy fall back to Agent[None, str].
        self._agent = Agent[None, list[ClaimDraft]](
            model=model,
            output_type=list[ClaimDraft],
            system_prompt=_PROMPT.get_system_prompt(),
            output_retries=output_retries,
        )

    async def run(
        self,
        llm_response: str,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> tuple[list[Claim], list[ExtractionWarning]]:
        """
        Extract atomic epistemic claims from the LLM response.

        Returns
        -------
        tuple[list[Claim], list[ExtractionWarning]]

        Raises
        ------
        ClaimExtractionFailure
        """
        user_message = _PROMPT.get_text(
            "message.md",
            goal=intent_context.goal,
            domain=intent_context.domain,
            action_class=intent_context.action_class,
            consequence_class=intent_context.consequence_class,
            actor=intent_context.actor or "Not specified",
            decision_window=intent_context.decision_window or "No constraint",
            llm_response=llm_response,
            fence=fence_for(llm_response),
            term_definitions=_format_dict(decomposed_prompt.term_definitions)
                or "None declared.",
            injected_facts=_format_list(decomposed_prompt.injected_facts)
                or "None.",
            active_constraints=_format_list(decomposed_prompt.all_constraints)
                or "None declared.",
            active_boundaries=_format_list(decomposed_prompt.all_boundaries)
                or "None declared.",
            undeclared_terms=_format_list(decomposed_prompt.undeclared_terms)
                or "None.",
        )

        try:
            result = await self._agent.run(user_message)
        except Exception as exc:
            raise ClaimExtractionFailure(
                message=(
                    "Claim extraction failed after "
                    f"{self._output_retries} retries. "
                    f"Underlying error: {exc}"
                ),
                attempts=self._output_retries,
                last_error=exc,
            ) from exc

        warnings: list[ExtractionWarning] = []
        raw_claims = _promote_drafts(result.output, warnings)

        raw_claims = _deduplicate_claim_ids(raw_claims, warnings)

        if self._verify_source_spans:
            raw_claims, span_warnings = _check_source_spans(
                raw_claims, llm_response
            )
            warnings.extend(span_warnings)

        return raw_claims, warnings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_list(items: list[str]) -> str:
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def _format_dict(mapping: dict[str, str]) -> str:
    if not mapping:
        return ""
    return "\n".join(f"- **{k}**: {v}" for k, v in mapping.items())


# ---------------------------------------------------------------------------
# Post-extraction validators
# ---------------------------------------------------------------------------


def _deduplicate_claim_ids(
    claims: list[Claim],
    warnings: list[ExtractionWarning],
) -> list[Claim]:
    seen: set[str] = set()
    result: list[Claim] = []
    for claim in claims:
        if claim.claim_id in seen:
            new_id = str(uuid.uuid4())
            warnings.append(ExtractionWarning(
                reason=(
                    f"Duplicate claim_id '{claim.claim_id}' "
                    f"reassigned to '{new_id}'."
                ),
                source_fragment=claim.source_span[:80],
            ))
            claim = claim.model_copy(update={"claim_id": new_id})
        seen.add(claim.claim_id)
        result.append(claim)
    return result


class ClaimDraft(Claim):
    """
    The shape the extractor LLM is asked to produce: a Claim whose
    cross-field invariants are NOT enforced at parse time.

    With ``output_type=list[Claim]`` a single structurally invalid claim
    (a causal claim with no mechanism, a prediction with no horizon) failed
    schema validation, exhausted the retries, and discarded every other valid
    claim in the response — while the StructuralGuard that exists to reject
    exactly those claims never got to run. Drafts are parsed leniently and
    promoted by ``_promote_drafts``; structural violations are carried forward
    so the guard can mark them inadmissible and the gate can reject them.
    """

    def causal_requires_assumptions(self):  # type: ignore[override]
        return self

    def predictive_requires_time_horizon(self):  # type: ignore[override]
        return self

    def normative_modality_consistency(self):  # type: ignore[override]
        return self

    def dependency_ids_not_self_referential(self):  # type: ignore[override]
        return self


def _promote_drafts(
    drafts: list[ClaimDraft],
    warnings: list[ExtractionWarning],
) -> list[Claim]:
    """
    Turn drafts into Claims.

    * Fully valid               -> a normal Claim.
    * Only cross-field invariant
      violated (error loc == ()) -> kept via model_construct and flagged, so
                                    the StructuralGuard rejects it visibly.
    * Any field-level error      -> excluded, with a warning.
    """
    claims: list[Claim] = []
    for draft in drafts:
        data = draft.model_dump()
        try:
            claims.append(Claim.model_validate(data))
        except ValidationError as exc:
            if all(err["loc"] == () for err in exc.errors()):
                warnings.append(ExtractionWarning(
                    reason=(
                        f"Claim '{draft.claim_id}' is structurally invalid "
                        f"({exc.errors()[0]['msg']}); carried to the "
                        f"StructuralGuard for rejection."
                    ),
                    source_fragment=draft.source_span[:80],
                ))
                claims.append(Claim.model_construct(**data))
            else:
                warnings.append(ExtractionWarning(
                    reason=(
                        f"Claim '{draft.claim_id}' excluded: invalid fields "
                        f"({'; '.join(str(e['loc']) for e in exc.errors())})."
                    ),
                    source_fragment=draft.source_span[:80],
                ))
    return claims


def _squash(text: str) -> str:
    return " ".join(text.split())


def _check_source_spans(
    claims: list[Claim],
    llm_response: str,
) -> tuple[list[Claim], list[ExtractionWarning]]:
    valid: list[Claim] = []
    warnings: list[ExtractionWarning] = []
    # Whitespace-insensitive: models routinely re-flow line breaks/indent
    # inside a quoted span; that is not fabrication.
    haystack = _squash(llm_response)
    for claim in claims:
        if _squash(claim.source_span) in haystack:
            valid.append(claim)
        else:
            warnings.append(ExtractionWarning(
                reason=(
                    f"Claim '{claim.claim_id}' excluded: "
                    f"source_span '{claim.source_span[:60]}' "
                    "not found verbatim in LLM response. "
                    "Possible fabrication."
                ),
                source_fragment=claim.source_span[:80],
            ))
    return valid, warnings