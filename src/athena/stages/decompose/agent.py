"""
stages.decompose.agent
======================
Step 0 — Prompt Decomposition.

Uses util.Prompt to load system.md (static, cached) and render
message.md (per-call, with $variable substitution) for the user turn.

PydanticAI API (pydantic-ai >= 0.0.20)
---------------------------------------
- output_type    (not result_type)
- result.output  (not result.data)
- output_retries controls schema-validation retry count
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

from athena.core.exceptions import DecompositionFailure
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import DomainInfo
from athena.util.prompt import Prompt, fence_for

_PROMPT = Prompt("decompose")


class DecomposeAgent:
    """
    Step 0 — decomposes a raw prompt into a structured DecomposedPrompt.

    Parameters
    ----------
    model:
        A pydantic_ai Model instance. The calling system provides this.
    output_retries:
        Times to retry on schema validation failure. Default: 3.
    """

    def __init__(self, model: Model, output_retries: int = 3) -> None:
        self._output_retries = output_retries
        # Explicit generic parameters force the type checker to resolve
        # Agent[None, DecomposedPrompt] correctly.
        # Without this, pyright/mypy fall back to Agent[None, str].
        self._agent = Agent[None, DecomposedPrompt](
            model=model,
            output_type=DecomposedPrompt,
            system_prompt=_PROMPT.get_system_prompt(),
            output_retries=output_retries,
        )

    async def run(
        self,
        prompt: str,
        domain_info: DomainInfo,
    ) -> DecomposedPrompt:
        """
        Decompose the raw prompt into a structured DecomposedPrompt.

        Raises
        ------
        DecompositionFailure
            If the agent exhausts retries or the provider errors.
        """
        user_message = _PROMPT.get_text(
            "message.md",
            domain=domain_info.description[:120],  # brief domain hint
            prompt=prompt,
            fence=fence_for(prompt),
            domain_description=domain_info.description,
            declared_constraints=_format_list(domain_info.constraints)
                or "None declared.",
            declared_boundaries=_format_list(domain_info.boundaries)
                or "None declared.",
            declared_term_definitions=_format_dict(domain_info.term_definitions)
                or "None declared.",
        )

        try:
            result = await self._agent.run(user_message)
        except Exception as exc:
            raise DecompositionFailure(
                message=(
                    "Prompt decomposition failed after "
                    f"{self._output_retries} retries. "
                    f"Underlying error: {exc}"
                ),
                attempts=self._output_retries,
                last_error=exc,
            ) from exc

        decomposed: DecomposedPrompt = result.output

        # Defensive pass-through: ensure declared fields are present
        # even if the model omitted them.
        updates: dict = {}
        if not decomposed.explicit_constraints and domain_info.constraints:
            updates["explicit_constraints"] = list(domain_info.constraints)
        if not decomposed.explicit_boundaries and domain_info.boundaries:
            updates["explicit_boundaries"] = list(domain_info.boundaries)

        # DomainInfo term definitions always win on collision
        merged = dict(decomposed.term_definitions)
        for term, defn in domain_info.term_definitions.items():
            merged[term] = defn
        if merged != decomposed.term_definitions:
            updates["term_definitions"] = merged

        if updates:
            decomposed = decomposed.model_copy(update=updates)

        return decomposed


# ---------------------------------------------------------------------------
# Formatting helpers for template substitution
# ---------------------------------------------------------------------------

def _format_list(items: list[str]) -> str:
    """Render a list as a numbered markdown list, or empty string."""
    if not items:
        return ""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _format_dict(mapping: dict[str, str]) -> str:
    """Render a dict as a bullet markdown list, or empty string."""
    if not mapping:
        return ""
    return "\n".join(f"- **{k}**: {v}" for k, v in mapping.items())