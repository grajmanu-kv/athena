"""
tests/test_decide.py
====================
Unit tests for the decision layer and dependency propagation.
No LLM calls.
"""

from __future__ import annotations

import pytest

from athena import derive_risk_level
from athena.core.models.claim import Claim, ClaimType, EpistemicModality
from athena.core.models.input import ActionClass, ConsequenceClass
from athena.core.models.result import (
    AdmissibilityState,
    ClaimResult,
    GuardOutcome,
    GuardResult,
    RiskLevel,
    VerdictCode,
)
from athena.core.protocols.dependency import DependencyGraph
from athena.stages.decide.gate import DecisionGate
from athena.stages.resolve.propagator import FailurePropagator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_claim_result(
    state: AdmissibilityState,
    claim_type: ClaimType = ClaimType.factual,
    modality: EpistemicModality = EpistemicModality.asserted,
    has_warning: bool = False,
    claim_id: str | None = None,
) -> ClaimResult:
    import uuid
    cid = claim_id or str(uuid.uuid4())
    guard_results = []
    if has_warning:
        guard_results.append(GuardResult(
            guard_id="TestGuard",
            outcome=GuardOutcome.warning,
            message="soft warning",
            detail={},
        ))
    return ClaimResult(
        claim=Claim.model_construct(
            claim_id=cid,
            claim_type=claim_type,
            epistemic_modality=modality,
            proposition="test proposition",
            source_span="test source span",
            scope="test scope",
            conditions=[],
            assumptions=["mechanism"] if claim_type == ClaimType.causal else [],
            depends_on=[],
            negates=[],
        ),
        guard_results=guard_results,
        admissibility_state=state,
        admissibility_rationale="test",
    )


# ---------------------------------------------------------------------------
# Risk matrix
# ---------------------------------------------------------------------------


class TestRiskMatrix:
    @pytest.mark.parametrize("action,consequence,expected", [
        (ActionClass.inform,    ConsequenceClass.negligible,  RiskLevel.low),
        (ActionClass.inform,    ConsequenceClass.critical,    RiskLevel.medium),
        (ActionClass.recommend, ConsequenceClass.critical,    RiskLevel.high),
        (ActionClass.decide,    ConsequenceClass.significant, RiskLevel.high),
        (ActionClass.automate,  ConsequenceClass.critical,    RiskLevel.critical),
        (ActionClass.automate,  ConsequenceClass.low,         RiskLevel.high),
    ])
    def test_matrix_entries(self, action, consequence, expected):
        assert derive_risk_level(action, consequence) == expected

    def test_all_16_defined(self):
        for ac in ActionClass:
            for cc in ConsequenceClass:
                risk = derive_risk_level(ac, cc)
                assert risk in RiskLevel


# ---------------------------------------------------------------------------
# DecisionGate
# ---------------------------------------------------------------------------


class TestDecisionGate:
    @pytest.fixture
    def gate(self):
        return DecisionGate()

    def test_critical_always_escalates(self, gate):
        claims = [make_claim_result(AdmissibilityState.verified)]
        v = gate.decide(claims, ActionClass.automate, ConsequenceClass.critical)
        assert v.code == VerdictCode.escalated
        assert v.risk_level == RiskLevel.critical

    def test_low_risk_all_plausible_accepted(self, gate):
        claims = [
            make_claim_result(AdmissibilityState.plausible),
            make_claim_result(AdmissibilityState.verified),
        ]
        v = gate.decide(claims, ActionClass.inform, ConsequenceClass.negligible)
        assert v.code == VerdictCode.accepted
        assert v.risk_level == RiskLevel.low

    def test_low_risk_with_warnings_accepted_with_warnings(self, gate):
        claims = [make_claim_result(AdmissibilityState.plausible, has_warning=True)]
        v = gate.decide(claims, ActionClass.inform, ConsequenceClass.negligible)
        assert v.code == VerdictCode.accepted_with_warnings

    def test_medium_risk_inadmissible_rejected(self, gate):
        claims = [
            make_claim_result(AdmissibilityState.plausible),
            make_claim_result(AdmissibilityState.inadmissible),
        ]
        v = gate.decide(claims, ActionClass.recommend, ConsequenceClass.low)
        assert v.code == VerdictCode.rejected
        assert len(v.inadmissible_claim_ids) == 1

    def test_medium_risk_contradicted_rejected(self, gate):
        claims = [make_claim_result(AdmissibilityState.contradicted)]
        v = gate.decide(claims, ActionClass.decide, ConsequenceClass.low)
        assert v.code == VerdictCode.rejected

    def test_high_risk_unsubstantiated_escalates(self, gate):
        claims = [make_claim_result(AdmissibilityState.unsubstantiated)]
        v = gate.decide(claims, ActionClass.decide, ConsequenceClass.significant)
        assert v.code == VerdictCode.escalated

    def test_high_risk_all_plausible_accepted(self, gate):
        claims = [make_claim_result(AdmissibilityState.plausible)]
        v = gate.decide(claims, ActionClass.decide, ConsequenceClass.significant)
        assert v.code == VerdictCode.accepted

    def test_verdict_tracks_inadmissible_ids(self, gate):
        bad_id = "bad-claim-id"
        claims = [make_claim_result(AdmissibilityState.inadmissible, claim_id=bad_id)]
        v = gate.decide(claims, ActionClass.inform, ConsequenceClass.low)
        assert bad_id in v.inadmissible_claim_ids

    def test_empty_claims_accepted_with_warnings(self, gate):
        v = gate.decide([], ActionClass.inform, ConsequenceClass.low)
        assert v.code == VerdictCode.accepted_with_warnings


# ---------------------------------------------------------------------------
# FailurePropagator
# ---------------------------------------------------------------------------


class TestFailurePropagator:
    @pytest.fixture
    def propagator(self):
        return FailurePropagator()

    def test_no_dependencies_no_change(self, propagator):
        cr = make_claim_result(AdmissibilityState.plausible)
        graph = DependencyGraph()
        results = propagator.propagate([cr], graph)
        assert results[0].admissibility_state == AdmissibilityState.plausible
        assert results[0].propagated_failure is False

    def test_inadmissible_upstream_propagates_hard(self, propagator):
        upstream = make_claim_result(
            AdmissibilityState.inadmissible, claim_id="upstream"
        )
        downstream = make_claim_result(
            AdmissibilityState.plausible, claim_id="downstream"
        )
        graph = DependencyGraph()
        graph.add_dependency("downstream", "upstream")

        results = propagator.propagate([upstream, downstream], graph)
        result_map = {r.claim.claim_id: r for r in results}

        assert result_map["upstream"].admissibility_state == AdmissibilityState.inadmissible
        assert result_map["downstream"].admissibility_state == AdmissibilityState.inadmissible
        assert result_map["downstream"].propagated_failure is True

    def test_contradicted_upstream_propagates_soft(self, propagator):
        upstream = make_claim_result(
            AdmissibilityState.contradicted, claim_id="upstream"
        )
        downstream = make_claim_result(
            AdmissibilityState.plausible, claim_id="downstream"
        )
        graph = DependencyGraph()
        graph.add_dependency("downstream", "upstream")

        results = propagator.propagate([upstream, downstream], graph)
        result_map = {r.claim.claim_id: r for r in results}

        # Soft: contradicted upstream → unsubstantiated downstream
        assert result_map["downstream"].admissibility_state == AdmissibilityState.unsubstantiated
        assert result_map["downstream"].propagated_failure is True

    def test_plausible_upstream_no_propagation(self, propagator):
        upstream = make_claim_result(
            AdmissibilityState.plausible, claim_id="upstream"
        )
        downstream = make_claim_result(
            AdmissibilityState.plausible, claim_id="downstream"
        )
        graph = DependencyGraph()
        graph.add_dependency("downstream", "upstream")

        results = propagator.propagate([upstream, downstream], graph)
        result_map = {r.claim.claim_id: r for r in results}

        assert result_map["downstream"].admissibility_state == AdmissibilityState.plausible
        assert result_map["downstream"].propagated_failure is False

    def test_negation_weaker_modality_downgraded(self, propagator):
        strong = make_claim_result(
            AdmissibilityState.plausible,
            modality=EpistemicModality.asserted,
            claim_id="strong",
        )
        weak = make_claim_result(
            AdmissibilityState.plausible,
            modality=EpistemicModality.speculative,
            claim_id="weak",
        )
        graph = DependencyGraph()
        graph.add_negation("strong", "weak")

        results = propagator.propagate([strong, weak], graph)
        result_map = {r.claim.claim_id: r for r in results}

        assert result_map["weak"].admissibility_state == AdmissibilityState.contradicted
        assert result_map["strong"].admissibility_state == AdmissibilityState.plausible

    def test_negation_equal_modality_both_contradicted(self, propagator):
        a = make_claim_result(
            AdmissibilityState.plausible,
            modality=EpistemicModality.asserted,
            claim_id="claim-a",
        )
        b = make_claim_result(
            AdmissibilityState.plausible,
            modality=EpistemicModality.asserted,
            claim_id="claim-b",
        )
        graph = DependencyGraph()
        graph.add_negation("claim-a", "claim-b")

        results = propagator.propagate([a, b], graph)
        result_map = {r.claim.claim_id: r for r in results}

        assert result_map["claim-a"].admissibility_state == AdmissibilityState.contradicted
        assert result_map["claim-b"].admissibility_state == AdmissibilityState.contradicted


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------


class TestDependencyGraph:
    def test_add_dependency(self):
        g = DependencyGraph()
        g.add_dependency("a", "b")
        assert "b" in g.dependencies_of("a")

    def test_add_negation_symmetric(self):
        g = DependencyGraph()
        g.add_negation("a", "b")
        assert "b" in g.negations_of("a")
        assert "a" in g.negations_of("b")

    def test_no_cycle_simple(self):
        g = DependencyGraph()
        g.add_dependency("b", "a")
        assert g.has_cycle() is False

    def test_cycle_detected(self):
        g = DependencyGraph()
        g.add_dependency("a", "b")
        g.add_dependency("b", "a")
        assert g.has_cycle() is True

    def test_topological_order(self):
        g = DependencyGraph()
        g.add_dependency("c", "b")
        g.add_dependency("b", "a")
        order = g.topological_order()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_topological_order_raises_on_cycle(self):
        from athena.core.exceptions import DependencyGraphCycleError
        g = DependencyGraph()
        g.add_dependency("a", "b")
        g.add_dependency("b", "a")
        with pytest.raises(DependencyGraphCycleError):
            g.topological_order()