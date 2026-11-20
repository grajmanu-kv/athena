"""
tests/test_models.py
====================
Unit tests for core Pydantic models.
No LLM calls. Pure validation logic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from athena import derive_risk_level
from athena.core.models.claim import (
    Claim,
    ClaimType,
    EpistemicModality,
    TimeHorizon,
)
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import (
    ActionClass,
    ConsequenceClass,
    DomainInfo,
)
from athena.core.models.result import (
    AdmissibilityState,
    ClaimResult,
    GuardOutcome,
    GuardResult,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# TimeHorizon
# ---------------------------------------------------------------------------


class TestTimeHorizon:
    def test_quantified_horizon(self):
        th = TimeHorizon(raw="within 15 minutes", normalized="PT15M")
        assert th.is_quantified is True

    def test_vague_horizon(self):
        th = TimeHorizon(raw="in the near future", normalized=None)
        assert th.is_quantified is False

    def test_none_string_coerced(self):
        th = TimeHorizon(raw="soon", normalized="None")
        assert th.normalized is None
        assert th.is_quantified is False

    def test_frozen(self):
        th = TimeHorizon(raw="PT15M", normalized="PT15M")
        with pytest.raises(Exception):
            th.raw = "changed"


# ---------------------------------------------------------------------------
# Claim model validators
# ---------------------------------------------------------------------------


class TestClaimValidators:
    def test_valid_factual_claim(self, factual_asserted_claim):
        assert factual_asserted_claim.claim_type == ClaimType.factual

    def test_causal_requires_assumptions(self):
        with pytest.raises(ValidationError, match="assumption"):
            Claim(
                claim_type=ClaimType.causal,
                epistemic_modality=EpistemicModality.asserted,
                proposition="X caused Y.",
                source_span="X caused Y.",
                scope="test",
                assumptions=[],  # missing — must fail
            )

    def test_causal_with_assumption_passes(self, causal_asserted_claim):
        assert len(causal_asserted_claim.assumptions) >= 1

    def test_predictive_requires_time_horizon(self):
        with pytest.raises(ValidationError, match="time_horizon"):
            Claim(
                claim_type=ClaimType.predictive,
                epistemic_modality=EpistemicModality.asserted,
                proposition="Traffic will worsen.",
                source_span="Traffic will worsen.",
                scope="Corridor A",
                time_horizon=None,
            )

    def test_predictive_with_horizon_passes(self, predictive_hedged_claim):
        assert predictive_hedged_claim.time_horizon is not None

    def test_normative_speculative_rejected(self):
        with pytest.raises(ValidationError, match="cannot have modality"):
            Claim(
                claim_type=ClaimType.normative,
                epistemic_modality=EpistemicModality.speculative,
                proposition="Maybe someone should do something.",
                source_span="Maybe someone should do something.",
                scope="test",
            )

    def test_normative_hypothetical_passes(self):
        c = Claim(
            claim_type=ClaimType.normative,
            epistemic_modality=EpistemicModality.hypothetical,
            proposition="If X, the operator should do Y.",
            source_span="If X, the operator should do Y.",
            scope="test",
        )
        assert c.epistemic_modality == EpistemicModality.hypothetical

    def test_claim_is_frozen(self, factual_asserted_claim):
        with pytest.raises(Exception):
            factual_asserted_claim.proposition = "mutated"

    def test_self_referential_depends_on_rejected(self):
        import uuid
        cid = str(uuid.uuid4())
        with pytest.raises(ValidationError, match="depends_on"):
            Claim(
                claim_id=cid,
                claim_type=ClaimType.factual,
                epistemic_modality=EpistemicModality.asserted,
                proposition="X.",
                source_span="X.",
                scope="test",
                depends_on=[cid],
            )


# ---------------------------------------------------------------------------
# DomainInfo
# ---------------------------------------------------------------------------


class TestDomainInfo:
    def test_has_explicit_contract(self, domain_information):
        assert domain_information.has_explicit_contract is True

    def test_no_contract(self):
        d = DomainInfo(description="no constraints", constraints=[], boundaries=[])
        assert d.has_explicit_contract is False

    def test_has_term_definitions(self, domain_information):
        assert domain_information.has_term_definitions is True

    def test_frozen(self, domain_information):
        with pytest.raises(Exception):
            domain_information.description = "mutated"


# ---------------------------------------------------------------------------
# DecomposedPrompt
# ---------------------------------------------------------------------------


class TestDecomposedPrompt:
    def test_all_constraints_merges(self):
        dp = DecomposedPrompt(
            explicit_constraints=["A"],
            implicit_constraints=["B"],
        )
        assert dp.all_constraints == ["A", "B"]

    def test_all_boundaries_merges(self):
        dp = DecomposedPrompt(
            explicit_boundaries=["X"],
            implicit_boundaries=["Y"],
        )
        assert dp.all_boundaries == ["X", "Y"]

    def test_has_any_contract_true(self):
        dp = DecomposedPrompt(explicit_constraints=["A"])
        assert dp.has_any_contract is True

    def test_has_any_contract_false(self):
        dp = DecomposedPrompt()
        assert dp.has_any_contract is False

    def test_has_undeclared_terms(self):
        dp = DecomposedPrompt(undeclared_terms=["ATCS", "RCP+"])
        assert dp.has_undeclared_terms is True

    def test_llm_response_field(self):
        dp = DecomposedPrompt(llm_response="The density is 94 veh/km.")
        assert dp.llm_response == "The density is 94 veh/km."


# ---------------------------------------------------------------------------
# ClaimResult computed properties
# ---------------------------------------------------------------------------


class TestClaimResult:
    def _make_result(self, state, guard_outcomes=None):
        guard_outcomes = guard_outcomes or []
        return ClaimResult(
            claim=Claim(
                claim_type=ClaimType.factual,
                epistemic_modality=EpistemicModality.asserted,
                proposition="X.",
                source_span="X.",
                scope="test",
            ),
            guard_results=[
                GuardResult(
                    guard_id="TestGuard",
                    outcome=o,
                    message="test",
                    detail={},
                )
                for o in guard_outcomes
            ],
            admissibility_state=state,
            admissibility_rationale="test",
        )

    def test_is_admissible_verified(self):
        r = self._make_result(AdmissibilityState.verified)
        assert r.is_admissible is True

    def test_is_admissible_plausible(self):
        r = self._make_result(AdmissibilityState.plausible)
        assert r.is_admissible is True

    def test_is_admissible_inadmissible(self):
        r = self._make_result(AdmissibilityState.inadmissible)
        assert r.is_admissible is False

    def test_has_warnings(self):
        r = self._make_result(
            AdmissibilityState.plausible,
            guard_outcomes=[GuardOutcome.passed, GuardOutcome.warning],
        )
        assert r.has_warnings is True

    def test_failed_guards(self):
        r = self._make_result(
            AdmissibilityState.inadmissible,
            guard_outcomes=[GuardOutcome.failed, GuardOutcome.passed],
        )
        assert len(r.failed_guards) == 1


# ---------------------------------------------------------------------------
# Risk matrix
# ---------------------------------------------------------------------------


class TestRiskMatrix:
    def test_all_16_combinations(self):
        for ac in ActionClass:
            for cc in ConsequenceClass:
                risk = derive_risk_level(ac, cc)
                assert isinstance(risk, RiskLevel)

    def test_automate_critical_is_critical(self):
        assert derive_risk_level(
            ActionClass.automate, ConsequenceClass.critical
        ) == RiskLevel.critical

    def test_inform_negligible_is_low(self):
        assert derive_risk_level(
            ActionClass.inform, ConsequenceClass.negligible
        ) == RiskLevel.low

    def test_recommend_critical_is_high(self):
        assert derive_risk_level(
            ActionClass.recommend, ConsequenceClass.critical
        ) == RiskLevel.high