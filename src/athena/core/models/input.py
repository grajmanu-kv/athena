"""
core.models.input
=================
The complete input contract for the Athena epistemic validation pipeline.

Four types form the input contract:

    ValidationInput     — the top-level object passed to EpistemicValidator.validate()
    IntentContext       — why this LLM call was made, what it feeds, who acts on it
    DomainInfo          — background knowledge, constraints, boundaries, term definitions
    ActionClass         — what category of action the output will feed
    ConsequenceClass    — what happens if the output is wrong

Design notes
------------
* IntentContext and DomainInfo are deliberately separate.
  IntentContext describes the *purpose* of this specific invocation.
  DomainInfo describes the *prompt contract* — what was required and
  forbidden of the LLM, and what background knowledge was provided.
  A different calling system with the same intent may declare different
  domain constraints.

* risk_level is NOT a direct input. It is derived from the
  ActionClass × ConsequenceClass matrix in stages/decide/risk_matrix.py.
  Consumers cannot override it directly — they can only influence it
  by declaring accurate action and consequence classes.

* constraints and boundaries live in DomainInfo, not IntentContext.
  They are properties of the prompt contract, not of the calling intent.

* known_facts feeds Step 4 (admissibility assessment).
  constraints and boundaries feed Step 3 (guard execution).
  term_definitions feed both Step 0 (augmentation) and Step 3 (terminology guard).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionClass(StrEnum):
    """
    What category of action this LLM output will feed.

    inform    : output is consumed as information; no direct action taken.
                e.g. a summary report read by an analyst.

    recommend : output is considered by a human as one input among many.
                e.g. a suggested course of action reviewed before deciding.

    decide    : output directly determines a binary or categorical decision.
                e.g. an automated triage that classifies an incident.

    automate  : output triggers automated system action with no human review.
                e.g. a signal controller receiving a timing recommendation
                that is applied without operator confirmation.
    """

    inform    = "inform"
    recommend = "recommend"
    decide    = "decide"
    automate  = "automate"


class ConsequenceClass(StrEnum):
    """
    What happens if the LLM output is wrong.

    negligible  : cosmetic or informational error; no material impact.

    low         : recoverable error; minor cost or inconvenience.

    significant : material cost, difficult to reverse, affects operations.

    critical    : safety, legal, financial, or irreversible consequence.
                  Always triggers rejection + mandatory human review in
                  the decision gate, regardless of claim admissibility.
    """

    negligible  = "negligible"
    low         = "low"
    significant = "significant"
    critical    = "critical"


# ---------------------------------------------------------------------------
# IntentContext
# ---------------------------------------------------------------------------


class IntentContext(BaseModel):
    """
    Declares why this LLM call was made and what it feeds.

    This is meta-intent: what the *calling system* knows about the purpose
    of this invocation that the LLM itself may not have been told.

    It is the most important input for the decision gate — it defines
    the stakes that determine what epistemic standard a claim must meet
    to be admissible.
    """

    model_config = {"frozen": True}

    goal: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "What the calling system is trying to accomplish. "
                "e.g. 'Recommend whether to trigger manual signal override "
                "on Corridor A'"
            ),
        ),
    ]

    actor: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Who will act on this output. "
                "e.g. 'Traffic operations officer', "
                "'Automated control system', 'End user'. "
                "None if not applicable or unknown."
            ),
        ),
    ] = None

    action_class: Annotated[
        ActionClass,
        Field(
            description=(
                "What category of action this output feeds. "
                "Together with consequence_class, determines risk level."
            ),
        ),
    ]

    consequence_class: Annotated[
        ConsequenceClass,
        Field(
            description=(
                "What happens if the output is wrong. "
                "Together with action_class, determines risk level. "
                "ConsequenceClass.critical always triggers rejection "
                "regardless of claim admissibility results."
            ),
        ),
    ]

    decision_window: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "How long the actor has to act on this output. "
                "Free text — not parsed by the library, used in audit trail "
                "and for time_horizon validation against operational urgency. "
                "e.g. '90 seconds', '24 hours', 'no constraint'"
            ),
        ),
    ] = None

    domain: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "High-level domain name. "
                "e.g. 'smart city traffic management', "
                "'medical diagnosis support', "
                "'financial advisory', 'legal document review'"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# DomainInfo
# ---------------------------------------------------------------------------


class DomainInfo(BaseModel):
    """
    Background knowledge about the domain and the prompt contract.

    Captures what the LLM was told, what it was required to do,
    what it was forbidden to do, and what background facts it was given.

    Pipeline routing:
        term_definitions → Step 0 (augmentation) + Step 3 (terminology guard)
        constraints      → Step 3 (constraint guard)
        boundaries       → Step 3 (boundary guard)
        known_facts      → Step 4 (admissibility assessment)
        description      → Step 0 + Step 1 (context for decomposition/extraction)
    """

    model_config = {"frozen": True}

    description: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Narrative description of the domain context. "
                "Provides the extraction and decomposition agents with "
                "the background needed to correctly classify claims. "
                "e.g. 'FusionWorld.AI smart city platform monitoring 47 intersections. "
                "Sensors sample every 30 seconds. Vehicle density in veh/km.'"
            ),
        ),
    ]

    term_definitions: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description=(
                "Locally defined terms that override general language semantics. "
                "Claims using these terms are validated against these definitions, "
                "not against everyday usage or LLM training distribution meanings. "
                "e.g. {'anomaly': 'sensor reading exceeding 3σ from 24h rolling mean', "
                "       'congestion': 'density > 80 veh/km sustained > 5 minutes'}"
            ),
        ),
    ]

    known_facts: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Ground truth facts available for admissibility checks. "
                "The admissibility assessor uses these to evaluate factual claims. "
                "e.g. ['Corridor A speed limit is 40 km/h', "
                "       'Peak hour is 08:00-10:00 and 17:00-19:00 IST']"
            ),
        ),
    ]

    constraints: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Positive obligations the LLM response was required to satisfy. "
                "The constraint guard checks each claim against these. "
                "e.g. ['Must cite sensor ID for every numerical claim', "
                "       'Recommendations must be actionable within operator authority']"
            ),
        ),
    ]

    boundaries: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Prohibitions the LLM response was explicitly forbidden to violate. "
                "The boundary guard checks each claim against these. "
                "A boundary violation is categorically different from an "
                "unsubstantiated claim — it is a contract breach. "
                "e.g. ['Must not infer causation from correlation', "
                "       'Must not recommend actions outside operator control']"
            ),
        ),
    ]

    @model_validator(mode="after")
    def warn_if_no_constraints_or_boundaries(self) -> "DomainInfo":
        """
        A DomainInfo with neither constraints nor boundaries means the
        constraint and boundary guards have nothing to check.
        This is valid but epistemically weak — the library cannot verify
        prompt contract compliance without a declared contract.

        Not an error — many prompts have implicit rather than explicit
        constraints. The decomposition stage will attempt to extract them.
        Logged as a diagnostic in the audit trail.
        """
        return self

    @property
    def has_explicit_contract(self) -> bool:
        """True when at least one constraint or boundary is declared."""
        return bool(self.constraints or self.boundaries)

    @property
    def has_term_definitions(self) -> bool:
        """True when at least one term definition is declared."""
        return bool(self.term_definitions)


# ---------------------------------------------------------------------------
# ValidationInput — top-level entry point
# ---------------------------------------------------------------------------


class ValidationInput(BaseModel):
    """
    The complete input contract for EpistemicValidator.validate().

    This is the single object a consuming system constructs and passes
    to the validator. All pipeline stages receive their inputs from this
    object via the internal PipelineContext — consuming systems never
    interact with pipeline internals directly.

    Example usage:

        from athena import (
            ValidationInput, IntentContext, DomainInfo,
            ActionClass, ConsequenceClass,
        )

        input = ValidationInput(
            prompt=\"\"\"
                You are a traffic operations advisor.
                Anomaly means a sensor reading exceeding 3σ from the 24h mean.
                Analyse the current sensor data and recommend whether to
                trigger a manual signal override on Corridor A.
                Do not infer causation from correlation.
            \"\"\",
            intent_context=IntentContext(
                goal="Recommend whether to trigger manual signal override",
                actor="Traffic operations officer",
                action_class=ActionClass.decide,
                consequence_class=ConsequenceClass.significant,
                decision_window="90 seconds",
                domain="smart city traffic management",
            ),
            domain_info=DomainInfo(
                description="FusionWorld.AI platform, 47 intersections, 30s sample rate.",
                term_definitions={
                    "anomaly": "sensor reading exceeding 3σ from 24h rolling mean",
                },
                known_facts=[
                    "Corridor A speed limit is 40 km/h",
                ],
                constraints=[
                    "Must cite sensor ID for every numerical claim",
                ],
                boundaries=[
                    "Must not infer causation from correlation",
                    "Must not recommend actions outside operator control",
                ],
            ),
            llm_response=(
                "Sensor CAM-042 shows vehicle density at 94 veh/km, "
                "which is an anomaly. This is likely caused by the "
                "earlier signal fault. I recommend triggering a manual override."
            ),
        )
    """

    model_config = {"frozen": True}

    prompt: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The raw prompt sent to the LLM. "
                "Treated as opaque text — decomposed in Step 0. "
                "May contain instructions, role framing, term definitions, "
                "output directives, and injected context fused together."
            ),
        ),
    ]

    intent_context: Annotated[
        IntentContext,
        Field(
            description=(
                "Why this LLM call was made and what it feeds. "
                "Drives risk level derivation and decision gate behaviour."
            ),
        ),
    ]

    domain_info: Annotated[
        DomainInfo,
        Field(
            description=(
                "Background knowledge, constraints, boundaries, "
                "and term definitions for this domain and prompt contract."
            ),
        ),
    ]

    llm_response: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The raw response from the LLM. "
                "Treated as opaque text — claims extracted from it in Step 1."
            ),
        ),
    ]