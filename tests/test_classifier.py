"""
tests/test_classifier.py
========================
Unit tests for the deterministic ClaimClassifier (Step 2).
No LLM calls.
"""

from __future__ import annotations

import pytest

from athena.core.models.claim import Claim, ClaimType, EpistemicModality
from athena.stages.classifier.agent import ClaimClassifier


@pytest.fixture
def classifier():
    return ClaimClassifier()


class TestTypeModalityMatrix:
    def test_normative_speculative_invalid(self, classifier):
        """normative + speculative is a category error."""
        # Pydantic blocks this at construction so we test classifier
        # via model_construct to bypass validation
        claim = Claim.model_construct(
            claim_id="test-001",
            claim_type=ClaimType.normative,
            epistemic_modality=EpistemicModality.speculative,
            proposition="Maybe someone should do something.",
            source_span="Maybe someone should do something.",
            scope="test",
            conditions=[],
            assumptions=[],
            depends_on=[],
            negates=[],
        )
        result = classifier.classify(claim)
        assert result.is_valid is False
        assert any("category error" in w.lower() for w in result.warnings)

    def test_factual_asserted_valid(self, classifier, factual_asserted_claim):
        result = classifier.classify(factual_asserted_claim)
        assert result.is_valid is True
        assert not result.warnings

    def test_causal_speculative_valid(self, classifier):
        """causal + speculative is valid — speculative causal claim."""
        claim = Claim(
            claim_type=ClaimType.causal,
            epistemic_modality=EpistemicModality.speculative,
            proposition="X might have caused Y.",
            source_span="X might have caused Y.",
            scope="test",
            assumptions=["X preceded Y"],
        )
        result = classifier.classify(claim)
        assert result.is_valid is True


class TestConfidenceLanguageConsistency:
    def test_asserted_with_hedging_warns(self, classifier):
        claim = Claim(
            claim_type=ClaimType.factual,
            epistemic_modality=EpistemicModality.asserted,
            proposition="Density is probably 94 veh/km.",
            source_span="Density is probably 94 veh/km.",
            scope="test",
            confidence_language="probably",
        )
        result = classifier.classify(claim)
        assert result.has_issues
        assert any("hedging" in w.lower() or "mismatch" in w.lower()
                   for w in result.warnings)

    def test_speculative_with_certainty_warns(self, classifier):
        claim = Claim(
            claim_type=ClaimType.factual,
            epistemic_modality=EpistemicModality.speculative,
            proposition="Density is certainly 94 veh/km.",
            source_span="Density is certainly 94 veh/km.",
            scope="test",
            confidence_language="certainly",
        )
        result = classifier.classify(claim)
        assert result.has_issues

    def test_hedged_with_hedging_no_warn(self, classifier, predictive_hedged_claim):
        result = classifier.classify(predictive_hedged_claim)
        # No confidence language mismatch — modality and language agree
        confidence_warnings = [
            w for w in result.warnings if "mismatch" in w.lower()
        ]
        assert not confidence_warnings


class TestStructuralPreconditions:
    def test_causal_without_assumptions_warns(self, classifier):
        claim = Claim.model_construct(
            claim_id="test-002",
            claim_type=ClaimType.causal,
            epistemic_modality=EpistemicModality.asserted,
            proposition="X caused Y.",
            source_span="X caused Y.",
            scope="test",
            conditions=[],
            assumptions=[],  # empty
            depends_on=[],
            negates=[],
        )
        result = classifier.classify(claim)
        assert any("assumption" in w.lower() for w in result.warnings)

    def test_predictive_without_horizon_warns(self, classifier):
        claim = Claim.model_construct(
            claim_id="test-003",
            claim_type=ClaimType.predictive,
            epistemic_modality=EpistemicModality.asserted,
            proposition="Traffic will worsen.",
            source_span="Traffic will worsen.",
            scope="test",
            conditions=[],
            assumptions=[],
            time_horizon=None,
            depends_on=[],
            negates=[],
        )
        result = classifier.classify(claim)
        assert any("time_horizon" in w.lower() for w in result.warnings)

    def test_predictive_vague_horizon_warns(
        self, classifier, predictive_vague_horizon_claim
    ):
        result = classifier.classify(predictive_vague_horizon_claim)
        assert any("vague" in w.lower() for w in result.warnings)

    def test_predictive_quantified_horizon_no_structural_warn(
        self, classifier, predictive_hedged_claim
    ):
        result = classifier.classify(predictive_hedged_claim)
        structural_warnings = [
            w for w in result.warnings
            if "time_horizon" in w.lower() or "vague" in w.lower()
        ]
        assert not structural_warnings


class TestScopeSpecificity:
    def test_unspecified_scope_warns_factual(self, classifier):
        claim = Claim(
            claim_type=ClaimType.factual,
            epistemic_modality=EpistemicModality.asserted,
            proposition="Density is high.",
            source_span="Density is high.",
            scope="unspecified scope — no location given",
        )
        result = classifier.classify(claim)
        assert any("scope" in w.lower() for w in result.warnings)

    def test_specific_scope_no_warn(self, classifier, factual_asserted_claim):
        result = classifier.classify(factual_asserted_claim)
        scope_warnings = [w for w in result.warnings if "scope" in w.lower()]
        assert not scope_warnings


class TestClassifyAll:
    def test_classify_all_returns_same_count(
        self,
        classifier,
        factual_asserted_claim,
        causal_asserted_claim,
        predictive_hedged_claim,
    ):
        claims = [
            factual_asserted_claim,
            causal_asserted_claim,
            predictive_hedged_claim,
        ]
        results = classifier.classify_all(claims)
        assert len(results) == 3
        assert results[0].claim_id == factual_asserted_claim.claim_id