"""
tests/test_guards.py
====================
Unit tests for built-in guards.

StructuralGuard: no LLM — tested directly.
ConstraintGuard, BoundaryGuard, TerminologyGuard: use pydantic_ai TestModel.
"""

from __future__ import annotations

import pytest

from athena.core.models.claim import Claim, ClaimType, EpistemicModality
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import ActionClass, ConsequenceClass, IntentContext
from athena.core.models.result import GuardOutcome
from athena.stages.guard.builtin.structural import StructuralGuard


# ---------------------------------------------------------------------------
# StructuralGuard — no LLM, test directly
# ---------------------------------------------------------------------------


@pytest.fixture
def structural_guard():
    return StructuralGuard()


@pytest.fixture
def empty_decomposed_prompt():
    return DecomposedPrompt()


@pytest.fixture
def basic_intent():
    return IntentContext(
        goal="test",
        action_class=ActionClass.inform,
        consequence_class=ConsequenceClass.low,
        domain="test",
    )


class TestStructuralGuard:
    @pytest.mark.asyncio
    async def test_passes_valid_factual(
        self, structural_guard, factual_asserted_claim,
        empty_decomposed_prompt, basic_intent
    ):
        result = await structural_guard.evaluate(
            factual_asserted_claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.passed

    @pytest.mark.asyncio
    async def test_fails_empty_source_span(
        self, structural_guard, empty_decomposed_prompt, basic_intent
    ):
        claim = Claim.model_construct(
            claim_id="test-sg-001",
            claim_type=ClaimType.factual,
            epistemic_modality=EpistemicModality.asserted,
            proposition="X.",
            source_span="",  # empty
            scope="test",
            conditions=[],
            assumptions=[],
            depends_on=[],
            negates=[],
        )
        result = await structural_guard.evaluate(
            claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.failed

    @pytest.mark.asyncio
    async def test_fails_predictive_no_horizon(
        self, structural_guard, empty_decomposed_prompt, basic_intent
    ):
        claim = Claim.model_construct(
            claim_id="test-sg-002",
            claim_type=ClaimType.predictive,
            epistemic_modality=EpistemicModality.asserted,
            proposition="Traffic will worsen.",
            source_span="Traffic will worsen.",
            scope="Corridor A",
            conditions=[],
            assumptions=[],
            time_horizon=None,
            depends_on=[],
            negates=[],
        )
        result = await structural_guard.evaluate(
            claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.failed

    @pytest.mark.asyncio
    async def test_warns_vague_time_horizon(
        self, structural_guard, predictive_vague_horizon_claim,
        empty_decomposed_prompt, basic_intent
    ):
        result = await structural_guard.evaluate(
            predictive_vague_horizon_claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.warning

    @pytest.mark.asyncio
    async def test_passes_causal_with_assumptions(
        self, structural_guard, causal_asserted_claim,
        empty_decomposed_prompt, basic_intent
    ):
        result = await structural_guard.evaluate(
            causal_asserted_claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.passed

    @pytest.mark.asyncio
    async def test_fails_causal_no_assumptions(
        self, structural_guard, empty_decomposed_prompt, basic_intent
    ):
        claim = Claim.model_construct(
            claim_id="test-sg-003",
            claim_type=ClaimType.causal,
            epistemic_modality=EpistemicModality.asserted,
            proposition="X caused Y.",
            source_span="X caused Y.",
            scope="test",
            conditions=[],
            assumptions=[],
            depends_on=[],
            negates=[],
        )
        result = await structural_guard.evaluate(
            claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.failed

    @pytest.mark.asyncio
    async def test_skips_when_no_constraints(
        self, structural_guard, factual_asserted_claim,
        basic_intent
    ):
        dp = DecomposedPrompt(explicit_constraints=[], implicit_constraints=[])
        result = await structural_guard.evaluate(
            factual_asserted_claim, dp, basic_intent
        )
        # StructuralGuard checks schema, not constraints — should pass
        assert result.outcome == GuardOutcome.passed

    @pytest.mark.asyncio
    async def test_warns_compound_proposition(
        self, structural_guard, empty_decomposed_prompt, basic_intent
    ):
        claim = Claim(
            claim_type=ClaimType.factual,
            epistemic_modality=EpistemicModality.asserted,
            proposition="Density is high and therefore congestion is severe.",
            source_span="Density is high and therefore congestion is severe.",
            scope="Corridor A",
        )
        result = await structural_guard.evaluate(
            claim, empty_decomposed_prompt, basic_intent
        )
        assert result.outcome == GuardOutcome.warning


# ---------------------------------------------------------------------------
# LLM-backed guards — use TestModel
# Tested for skip conditions (no LLM call needed) only in unit tests.
# Integration behaviour tested via test_manual_run.py.
# ---------------------------------------------------------------------------


class TestConstraintGuardSkipConditions:
    @pytest.mark.asyncio
    async def test_skips_when_no_constraints(
        self, factual_asserted_claim, basic_intent
    ):
        from pydantic_ai.models.test import TestModel
        from athena.stages.guard.builtin.constraint import ConstraintGuard

        guard = ConstraintGuard(model=TestModel())
        dp = DecomposedPrompt()  # no constraints
        result = await guard.evaluate(factual_asserted_claim, dp, basic_intent)
        assert result.outcome == GuardOutcome.skipped

    @pytest.mark.asyncio
    async def test_skips_normative_claims(
        self, normative_asserted_claim, basic_intent
    ):
        from pydantic_ai.models.test import TestModel
        from athena.stages.guard.builtin.constraint import ConstraintGuard

        guard = ConstraintGuard(model=TestModel())
        dp = DecomposedPrompt(explicit_constraints=["Must cite sensor ID."])
        result = await guard.evaluate(normative_asserted_claim, dp, basic_intent)
        assert result.outcome == GuardOutcome.skipped


class TestBoundaryGuardSkipConditions:
    @pytest.mark.asyncio
    async def test_skips_when_no_boundaries(
        self, factual_asserted_claim, basic_intent
    ):
        from pydantic_ai.models.test import TestModel
        from athena.stages.guard.builtin.boundary import BoundaryGuard

        guard = BoundaryGuard(model=TestModel())
        dp = DecomposedPrompt()  # no boundaries
        result = await guard.evaluate(factual_asserted_claim, dp, basic_intent)
        assert result.outcome == GuardOutcome.skipped


class TestTerminologyGuardSkipConditions:
    @pytest.mark.asyncio
    async def test_skips_when_no_definitions(
        self, factual_asserted_claim, basic_intent
    ):
        from pydantic_ai.models.test import TestModel
        from athena.stages.guard.builtin.terminology import TerminologyGuard

        guard = TerminologyGuard(model=TestModel())
        dp = DecomposedPrompt()  # no term_definitions, no undeclared_terms
        result = await guard.evaluate(factual_asserted_claim, dp, basic_intent)
        assert result.outcome == GuardOutcome.skipped