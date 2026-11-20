"""
Offline demo — runs the whole Athena pipeline with NO API key.

The LLM stages (decompose, extract, and the constraint / boundary /
terminology guards) are answered by a scripted stand-in model, so you can see
the real prompts, the real guards, assessors, dependency resolution and
decision gate working end to end:

    uv run python main.py

To run against a real model, build the validator with any pydantic-ai Model:

    from pydantic_ai.models.anthropic import AnthropicModel
    validator = EpistemicValidator(model=AnthropicModel("..."))
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from athena import (
    ActionClass, ConsequenceClass, DomainInfo, EpistemicValidator,
    IntentContext, ValidationInput,
)

RESPONSE = (
    "Sensor CAM-042 reports 87 veh/km on Corridor A. "
    "Density will likely exceed 100 veh/km within 15 minutes. "
    "The operator should notify the duty officer."
)

CLAIMS = [
    dict(claim_type="factual", epistemic_modality="asserted",
         proposition="Sensor CAM-042 reports 87 veh/km on Corridor A",
         source_span="Sensor CAM-042 reports 87 veh/km on Corridor A", scope="Corridor A, now"),
    dict(claim_type="predictive", epistemic_modality="hedged",
         proposition="Density on Corridor A will exceed 100 veh/km",
         source_span="Density will likely exceed 100 veh/km within 15 minutes",
         scope="Corridor A, next 15 minutes", confidence_language="likely",
         time_horizon=dict(raw="within 15 minutes", normalized="PT15M")),
    dict(claim_type="normative", epistemic_modality="asserted",
         proposition="The operator should notify the duty officer",
         source_span="The operator should notify the duty officer", scope="Corridor A, now"),
]


def scripted_model() -> FunctionModel:
    def respond(messages, info):
        tool = info.output_tools[0]
        props = tool.parameters_json_schema.get("properties", {})
        if "response" in props:
            args = {"response": CLAIMS}
        elif "implicit_constraints" in props:
            args = {"implicit_constraints": [], "implicit_boundaries": [],
                    "term_definitions": {}, "undeclared_terms": [], "injected_facts": []}
        elif "satisfies" in props:
            args = {"satisfies": True, "reasoning": "ok"}
        elif "violates" in props:
            args = {"violates": False, "reasoning": "ok"}
        else:
            args = {"consistent": True, "reasoning": "ok"}
        return ModelResponse(parts=[ToolCallPart(tool.name, args)])

    respond.__name__ = "scripted"
    return FunctionModel(respond)


async def main() -> None:
    validator = EpistemicValidator(model=scripted_model(), include_audit_report=True)
    result = await validator.validate(ValidationInput(
        prompt="Summarise Corridor A status.",
        intent_context=IntentContext(
            goal="Decide whether to notify the duty officer",
            action_class=ActionClass.recommend,
            consequence_class=ConsequenceClass.significant,
            domain="traffic operations",
            decision_window="30 minutes",
        ),
        domain_info=DomainInfo(
            description="Urban traffic operations",
            constraints=["Must cite a sensor ID for numerical claims."],
            boundaries=["Must not infer causation from correlation."],
        ),
        llm_response=RESPONSE,
    ))
    print(f"verdict: {result.verdict.code.value}  (risk: {result.verdict.risk_level.value})")
    print(result.verdict.rationale)
    for cr in result.claim_results:
        print(f"  [{cr.admissibility_state.value:>15}] {cr.claim.claim_type.value:<10} {cr.claim.proposition}")
    for w in result.pipeline_warnings:
        print("  warning:", w)


if __name__ == "__main__":
    asyncio.run(main())
