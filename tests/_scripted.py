"""
tests/_scripted.py
==================
A scripted stand-in for the LLM, used by the end-to-end regression tests.

pydantic_ai's FunctionModel calls `Script` for every agent call; the output
tool's JSON schema tells us which stage is asking, so we can answer each stage
deterministically (and, optionally, make the three LLM guards fail).
"""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from athena import EpistemicValidator
from athena.core.models.input import (
    ActionClass, ConsequenceClass, DomainInfo, IntentContext, ValidationInput,
)

RESPONSE = (
    "Sensor CAM-042 reports 87 veh/km on Corridor A. "
    "Density will likely exceed 100 veh/km within 15 minutes. "
    "The operator should notify the duty officer."
)


def good_claims() -> list[dict]:
    return [
        dict(claim_type="factual", epistemic_modality="asserted",
             proposition="Sensor CAM-042 reports 87 veh/km on Corridor A",
             source_span="Sensor CAM-042 reports 87 veh/km on Corridor A",
             scope="Corridor A, now"),
        dict(claim_type="predictive", epistemic_modality="hedged",
             proposition="Density on Corridor A will exceed 100 veh/km",
             source_span="Density will likely exceed 100 veh/km within 15 minutes",
             scope="Corridor A, next 15 minutes", confidence_language="likely",
             time_horizon=dict(raw="within 15 minutes", normalized="PT15M")),
        dict(claim_type="normative", epistemic_modality="asserted",
             proposition="The operator should notify the duty officer",
             source_span="The operator should notify the duty officer",
             scope="Corridor A, now"),
    ]


class Script:
    def __init__(self, claims: list[dict], fail_guards: bool = False) -> None:
        self.claims = claims
        self.fail_guards = fail_guards
        self.calls: list[str] = []
        self.__name__ = "script"          # FunctionModel needs a named callable

    def __call__(self, messages, info):
        tool = info.output_tools[0]
        props = tool.parameters_json_schema.get("properties", {})
        if "response" in props:
            kind, args = "extract", {"response": self.claims}
        elif "implicit_constraints" in props:
            kind, args = "decompose", {
                "implicit_constraints": [], "implicit_boundaries": [],
                "term_definitions": {}, "undeclared_terms": [], "injected_facts": [],
            }
        elif {"satisfies", "violates", "consistent"} & set(props):
            kind = "guard"
            if self.fail_guards:
                raise RuntimeError("simulated provider outage")
            args = ({"satisfies": True} if "satisfies" in props
                    else {"violates": False} if "violates" in props
                    else {"consistent": True})
            args["reasoning"] = "ok"
        else:
            kind, args = "unknown", {}
        self.calls.append(kind)
        # Outage applies to guards only: decompose/extract must still work.
        return ModelResponse(parts=[ToolCallPart(tool.name, args)])


def make_validator(script: Script, audit: bool = False) -> EpistemicValidator:
    return EpistemicValidator(model=FunctionModel(script), include_audit_report=audit)


def make_input(a=ActionClass.recommend, c=ConsequenceClass.significant,
               response: str = RESPONSE) -> ValidationInput:
    return ValidationInput(
        prompt="Summarise Corridor A status.",
        intent_context=IntentContext(goal="g", action_class=a, consequence_class=c,
                                     domain="traffic"),
        domain_info=DomainInfo(
            description="Traffic ops",
            constraints=["Must cite sensor ID for numerical claims."],
            boundaries=["Must not infer causation from correlation."],
            term_definitions={"congestion": "density > 80 veh/km"},
        ),
        llm_response=response,
    )
