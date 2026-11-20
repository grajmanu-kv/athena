"""athena.web.schemas: HTTP request model for /api/validate."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from athena import ActionClass, ConsequenceClass


def _clean_list(values: list[str]) -> list[str]:
    return [v.strip() for v in values if v and v.strip()]


class ValidateRequest(BaseModel):
    # The two things being tested
    query: str = Field(min_length=1, description="The prompt sent to the LLM.")
    response: str = Field(min_length=1, description="The LLM's answer to validate.")

    # Risk context (drives how strict the gate is)
    goal: str = Field(min_length=1, description="What the answer will be used for.")
    action_class: ActionClass
    consequence_class: ConsequenceClass
    actor: str | None = None
    decision_window: str | None = None
    domain: str = Field(default="general", min_length=1)

    # Domain contract (all optional)
    description: str = ""
    term_definitions: dict[str, str] = Field(default_factory=dict)
    known_facts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)

    # Optional per-request model override (Gemini model id)
    model: str | None = None

    @field_validator("query", "response", "goal", "domain")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("actor", "decision_window", "model")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        return (v or "").strip() or None

    @field_validator("known_facts", "constraints", "boundaries")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        return _clean_list(v)

    @field_validator("term_definitions")
    @classmethod
    def _clean_terms(cls, v: dict[str, str]) -> dict[str, str]:
        return {k.strip(): d.strip() for k, d in v.items() if k.strip() and d.strip()}
