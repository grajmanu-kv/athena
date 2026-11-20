"""
tests/conftest.py
=================
Shared pytest fixtures for Athena unit tests.

All tests use pydantic_ai.models.test.TestModel — no real API calls.
TestModel returns configurable structured outputs for each agent call.
"""

from __future__ import annotations


import pytest


from athena.core.models.claim import Claim, ClaimType, EpistemicModality, TimeHorizon
from athena.core.models.input import (
    ActionClass,
    ConsequenceClass,
    DomainInfo,
    IntentContext,
)


# ---------------------------------------------------------------------------
# Shared domain fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def domain_information() -> DomainInfo:
    return DomainInfo(
        description="FusionWorld.AI smart city platform. Sensors sample every 30s.",
        term_definitions={
            "anomaly": "sensor reading exceeding 3σ from 24h rolling mean",
            "congestion": "density > 80 veh/km sustained > 5 minutes",
        },
        known_facts=[
            "Corridor A speed limit is 40 km/h.",
            "Sensor CAM-042 monitors Corridor A.",
        ],
        constraints=[
            "Must cite sensor ID for every numerical claim.",
        ],
        boundaries=[
            "Must not infer causation from correlation.",
            "Must not recommend actions outside operator control.",
        ],
    )


@pytest.fixture
def recommend_low_intent() -> IntentContext:
    return IntentContext(
        goal="Recommend whether to notify traffic officer.",
        actor="Traffic operations officer",
        action_class=ActionClass.recommend,
        consequence_class=ConsequenceClass.low,
        decision_window="10 minutes",
        domain="smart city traffic management",
    )


@pytest.fixture
def automate_critical_intent() -> IntentContext:
    return IntentContext(
        goal="Automated signal control recommendation.",
        actor="Automated signal controller",
        action_class=ActionClass.automate,
        consequence_class=ConsequenceClass.critical,
        decision_window="30 seconds",
        domain="smart city traffic management",
    )


@pytest.fixture
def decide_significant_intent() -> IntentContext:
    return IntentContext(
        goal="Determine root cause of congestion.",
        actor="Traffic operations engineer",
        action_class=ActionClass.decide,
        consequence_class=ConsequenceClass.significant,
        decision_window="30 minutes",
        domain="smart city traffic management",
    )


# ---------------------------------------------------------------------------
# Claim fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factual_asserted_claim() -> Claim:
    return Claim(
        claim_type=ClaimType.factual,
        epistemic_modality=EpistemicModality.asserted,
        proposition="Vehicle density on Corridor A is 94 veh/km.",
        source_span="Vehicle density on Corridor A is 94 veh/km.",
        scope="Corridor A, 14:32 IST",
    )


@pytest.fixture
def causal_asserted_claim() -> Claim:
    return Claim(
        claim_type=ClaimType.causal,
        epistemic_modality=EpistemicModality.asserted,
        proposition="The signal fault caused the congestion on Corridor A.",
        source_span="The signal fault caused the congestion.",
        scope="Corridor A, 14:00-14:35 IST",
        assumptions=["signal fault preceded congestion onset by 8 minutes"],
    )


@pytest.fixture
def predictive_hedged_claim() -> Claim:
    return Claim(
        claim_type=ClaimType.predictive,
        epistemic_modality=EpistemicModality.hedged,
        proposition="Density will likely exceed 100 veh/km within 15 minutes.",
        source_span="Density will likely exceed 100 veh/km within 15 minutes.",
        scope="Corridor A, next 15 minutes from 14:35 IST",
        time_horizon=TimeHorizon(raw="within 15 minutes", normalized="PT15M"),
        confidence_language="likely",
    )


@pytest.fixture
def predictive_vague_horizon_claim() -> Claim:
    return Claim(
        claim_type=ClaimType.predictive,
        epistemic_modality=EpistemicModality.speculative,
        proposition="Conditions may improve in the near future.",
        source_span="Conditions may improve in the near future.",
        scope="Corridor A",
        time_horizon=TimeHorizon(raw="in the near future", normalized=None),
        confidence_language="may",
    )


@pytest.fixture
def normative_asserted_claim() -> Claim:
    return Claim(
        claim_type=ClaimType.normative,
        epistemic_modality=EpistemicModality.asserted,
        proposition="The operator should trigger a manual override.",
        source_span="The operator should trigger a manual override.",
        scope="Corridor A",
    )