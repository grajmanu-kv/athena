"""
tests/test_regressions.py
=========================
One test (or a few) per defect fixed in the v0.1.1 hardening pass. The
identifiers (P02, Q03 ...) refer to the validation report.

No network: LLM stages are driven by pydantic_ai's FunctionModel with a
scripted responder (see tests/_scripted.py).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from athena.core.models.claim import (
    Claim, ClaimType, EpistemicModality, TimeHorizon,
)
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import (
    ActionClass, ConsequenceClass, DomainInfo, IntentContext, ValidationInput,
)
from athena.core.models.result import (
    AdmissibilityState as S, ClaimResult, GuardOutcome, GuardResult, VerdictCode,
)
from athena.core.protocols.admissibility import AdmissibilityAssessment
from athena.core.protocols.dependency import DependencyGraph
from athena.stages.assess.assessor import AdmissibilityAssessor
from athena.stages.assess.builtin.structural_admissibility import (
    StructuralAdmissibilityChecker,
)
from athena.stages.decide.gate import DecisionGate
from athena.stages.extract.agent import ClaimDraft, _check_source_spans, _promote_drafts
from athena.stages.guard.builtin.structural import StructuralGuard
from athena.stages.guard.runner import GuardRunner
from athena.stages.resolve.builder import (
    HeuristicDependencyBuilder, _propositions_negate, _scopes_overlap,
)
from athena.stages.resolve.propagator import FailurePropagator
from athena.util.duration import parse_duration_seconds
from athena.util.prompt import fence_for

from tests._scripted import Script, good_claims, make_validator, make_input


# ---------------------------------------------------------------- helpers
def claim(ctype=ClaimType.factual, mod=EpistemicModality.asserted,
          prop="Density is 87 veh/km", scope="Corridor A, now", **kw):
    if ctype == ClaimType.causal and "assumptions" not in kw:
        kw["assumptions"] = ["mechanism"]
    if ctype == ClaimType.predictive and "time_horizon" not in kw:
        kw["time_horizon"] = TimeHorizon(raw="in 15 minutes", normalized="PT15M")
    return Claim(claim_type=ctype, epistemic_modality=mod, proposition=prop,
                 source_span=prop, scope=scope, **kw)


def cr(c, state=S.plausible, guards=(), **kw):
    return ClaimResult(claim=c, guard_results=list(guards), admissibility_state=state,
                       admissibility_rationale="t", **kw)


def gr(gid, outcome, msg="m"):
    return GuardResult(guard_id=gid, outcome=outcome, message=msg)


def intent(a=ActionClass.recommend, c=ConsequenceClass.low, window=None):
    return IntentContext(goal="g", action_class=a, consequence_class=c,
                         domain="d", decision_window=window)


DP = DecomposedPrompt()


# ================================================================ models
class TestModelFixes:
    @pytest.mark.parametrize("v", ["P", "PT", "P1DT", "P3DT", "T5M", "PT1H2"])
    def test_p07_empty_or_dangling_iso_duration_rejected(self, v):
        with pytest.raises(ValidationError):
            TimeHorizon(raw="x", normalized=v)

    @pytest.mark.parametrize("v", ["PT15M", "P1D", "P1W", "PT1H30M", "P1Y2M3DT4H5M6S", "P2W3D", "PT90S"])
    def test_valid_iso_durations_still_accepted(self, v):
        assert TimeHorizon(raw="x", normalized=v).is_quantified

    def test_p08_blank_assumption_does_not_satisfy_causal_invariant(self):
        with pytest.raises(ValidationError):
            claim(ClaimType.causal, assumptions=["   ", ""])

    def test_assumptions_are_stripped(self):
        c = claim(ClaimType.causal, assumptions=["  pipe is blocked  "])
        assert c.assumptions == ["pipe is blocked"]

    def test_checker_warnings_count_towards_has_warnings(self):
        assert cr(claim(), warnings=["w"]).has_warnings
        assert cr(claim(), guards=[gr("G", GuardOutcome.errored)]).has_warnings
        assert not cr(claim()).has_warnings


# ================================================================ guards
class TestGuardErrors:
    @pytest.mark.asyncio
    async def test_p17_guard_exception_is_errored_not_failed(self):
        class Boom:
            guard_id = "Boom"
            async def evaluate(self, *a, **k):
                raise RuntimeError("x")
        recs = await GuardRunner([Boom()]).run_all([claim()], DP, intent())
        out = recs[0].guard_results[0]
        assert out.outcome == GuardOutcome.errored
        assert recs[0].has_warnings and not recs[0].has_failures

    @pytest.mark.asyncio
    async def test_runner_preserves_claim_order_under_concurrency(self):
        import asyncio
        class Slow:
            guard_id = "Slow"
            async def evaluate(self, c, *a, **k):
                await asyncio.sleep(0.02 if c.proposition.endswith("0") else 0)
                return gr("Slow", GuardOutcome.passed, c.proposition)
        cs = [claim(prop=f"Claim number {i}") for i in range(10)]
        recs = await GuardRunner([Slow()]).run_all(cs, DP, intent())
        assert [r.claim_id for r in recs] == [c.claim_id for c in cs]

    def test_structural_guard_warns_when_horizon_exceeds_decision_window(self):
        c = claim(ClaimType.predictive,
                  time_horizon=TimeHorizon(raw="in 2 hours", normalized="PT2H"))
        import asyncio
        r = asyncio.run(StructuralGuard().evaluate(c, DP, intent(window="10 minutes")))
        assert r.outcome == GuardOutcome.warning and "decision window" in r.message

    def test_horizon_within_window_or_unparseable_window_passes(self):
        import asyncio
        c = claim(ClaimType.predictive)
        for w in ("1 hour", "no constraint", None):
            r = asyncio.run(StructuralGuard().evaluate(c, DP, intent(window=w)))
            assert r.outcome == GuardOutcome.passed

    @pytest.mark.parametrize("text,secs", [
        ("90 seconds", 90), ("24 hours", 86400), ("PT15M", 900), ("P1DT2H", 93600),
        ("2 days", 172800), ("15 min", 900), ("no constraint", None), (None, None), ("P", None),
    ])
    def test_parse_duration_seconds(self, text, secs):
        assert parse_duration_seconds(text) == secs


# ============================================================== assessors
class TestAssessment:
    @pytest.mark.asyncio
    async def test_f_order_independent_most_severe_failure_wins(self):
        c = claim()
        t = gr("TerminologyGuard", GuardOutcome.failed)
        s = gr("StructuralGuard", GuardOutcome.failed)
        for order in ([t, s], [s, t]):
            res = await StructuralAdmissibilityChecker().assess(c, [cr(c, guards=order)], DP, intent())
            assert res.state == S.inadmissible

    @pytest.mark.asyncio
    async def test_p04_failed_consumer_guard_is_not_ignored(self):
        c = claim()
        res = await StructuralAdmissibilityChecker().assess(
            c, [cr(c, guards=[gr("MyOwnGuard", GuardOutcome.failed)])], DP, intent())
        assert res.state == S.contradicted

    @pytest.mark.asyncio
    async def test_p17_errored_guard_is_unsubstantiated(self):
        c = claim()
        res = await StructuralAdmissibilityChecker().assess(
            c, [cr(c, guards=[gr("ConstraintGuard", GuardOutcome.errored)])], DP, intent())
        assert res.state == S.unsubstantiated

    def test_p15_internal_flag_no_longer_bypasses_reserved_state(self):
        with pytest.raises(ValueError):
            AdmissibilityAssessment("X", S.inadmissible, "r", _internal=True)

    @pytest.mark.asyncio
    async def test_p03_verified_is_reachable(self):
        class Oracle:
            checker_id = "Oracle"
            async def assess(self, *a, **k):
                return AdmissibilityAssessment("Oracle", S.verified, "matches known fact")
        c = claim()
        [res] = await AdmissibilityAssessor([Oracle()]).assess_all([c], {}, DP, intent())
        assert res.admissibility_state == S.verified

    @pytest.mark.asyncio
    async def test_verified_never_masks_a_worse_state(self):
        class Oracle:
            checker_id = "Oracle"
            async def assess(self, *a, **k):
                return AdmissibilityAssessment("Oracle", S.verified, "ok")
        c = claim(mod=EpistemicModality.speculative)           # high risk -> unsubstantiated
        [res] = await AdmissibilityAssessor([Oracle()]).assess_all(
            [c], {}, DP, intent(ActionClass.decide, ConsequenceClass.significant))
        assert res.admissibility_state == S.unsubstantiated

    @pytest.mark.asyncio
    async def test_foreign_inadmissible_assessment_is_demoted(self):
        class Rogue:
            checker_id = "Rogue"
            async def assess(self, *a, **k):
                class Fake:   # a duck-typed object that claims `inadmissible`
                    checker_id = "Rogue"; state = S.inadmissible
                    rationale = "r"; evidence = {}; warnings = []; _internal = False
                return Fake()
        [res] = await AdmissibilityAssessor([Rogue()]).assess_all([claim()], {}, DP, intent())
        assert res.admissibility_state == S.contradicted

    @pytest.mark.asyncio
    async def test_p20_modality_warning_reaches_the_gate(self):
        c = claim(mod=EpistemicModality.speculative)
        [res] = await AdmissibilityAssessor().assess_all(
            [c], {}, DP, intent(ActionClass.decide, ConsequenceClass.low))      # medium
        assert res.admissibility_state == S.plausible and res.has_warnings
        v = DecisionGate().decide([res], ActionClass.decide, ConsequenceClass.low)
        assert v.code == VerdictCode.accepted_with_warnings
        assert res.claim.claim_id in v.warning_claim_ids


# ================================================================== gate
class TestGate:
    @pytest.mark.parametrize("a,c,code", [
        (ActionClass.automate, ConsequenceClass.critical, VerdictCode.escalated),
        (ActionClass.decide, ConsequenceClass.significant, VerdictCode.escalated),   # high
        (ActionClass.decide, ConsequenceClass.low, VerdictCode.accepted_with_warnings),  # medium
        (ActionClass.inform, ConsequenceClass.negligible, VerdictCode.accepted_with_warnings),
    ])
    def test_p16_no_claims_is_never_a_clean_accept(self, a, c, code):
        assert DecisionGate().decide([], a, c).code == code

    def test_p02_contradicted_rejected_at_low_risk(self):
        v = DecisionGate().decide([cr(claim(), S.contradicted)],
                                  ActionClass.inform, ConsequenceClass.negligible)
        assert v.code == VerdictCode.rejected


# ============================================================== resolve
class TestResolve:
    def test_p05_two_compatible_should_not_claims_do_not_negate(self):
        assert not _propositions_negate("operator should not override the signal",
                                        "operator should not touch the controller")

    def test_should_vs_should_not_still_negate(self):
        assert _propositions_negate("operator should override", "operator should not override")

    def test_is_high_not_matched_inside_is_higher(self):
        assert not _propositions_negate("flow is higher than usual", "flow is low")

    @pytest.mark.parametrize("a,b,expected", [
        ("A, 14:00", "Airport Road, 14:00", False),          # P10: substring false positive
        ("Corridor A", "Corridor AB", False),
        ("Corridor A", "Corridor B", False),
        ("Corridor A, now", "Corridor A, 14:00", True),
        ("Corridor", "Corridor A, now", True),
    ])
    def test_p10_scope_overlap_is_word_based(self, a, b, expected):
        assert _scopes_overlap(a, b) is expected

    @pytest.mark.asyncio
    async def test_negation_requires_overlapping_scope(self):
        a = claim(prop="Density is high", scope="Corridor A, now")
        b = claim(prop="Density is low", scope="Corridor B, now")
        g = await HeuristicDependencyBuilder().build([cr(a), cr(b)], DP, intent())
        assert not g.negations_of(a.claim_id)
        c = claim(prop="Density is low", scope="Corridor A, now")
        g = await HeuristicDependencyBuilder().build([cr(a), cr(c)], DP, intent())
        assert c.claim_id in g.negations_of(a.claim_id)

    def test_p06_negation_never_overwrites_inadmissible(self):
        weak = claim(mod=EpistemicModality.speculative, prop="Flow is high")
        strong = claim(prop="Flow is low")
        g = DependencyGraph(); g.add_negation(weak.claim_id, strong.claim_id)
        out = {x.claim.claim_id: x for x in FailurePropagator().propagate(
            [cr(weak, S.inadmissible), cr(strong, S.plausible)], g)}
        assert out[weak.claim_id].admissibility_state == S.inadmissible

    def test_depends_on_and_negates_are_populated(self):
        f = claim(); k = claim(ClaimType.causal)
        g = DependencyGraph(); g.add_dependency(k.claim_id, f.claim_id)
        out = {x.claim.claim_id: x for x in FailurePropagator().propagate([cr(f), cr(k)], g)}
        assert out[k.claim_id].claim.depends_on == [f.claim_id]

    def test_cyclic_graph_raises_instead_of_failing_open(self):
        a, b = claim(), claim(prop="Other fact here")
        g = DependencyGraph()
        g.add_dependency(a.claim_id, b.claim_id)
        g.add_dependency(b.claim_id, a.claim_id)
        assert g.has_cycle()
        with pytest.raises(Exception):
            FailurePropagator().propagate([cr(a, S.inadmissible), cr(b)], g)

    def test_hard_failure_still_propagates(self):
        f = claim(); k = claim(ClaimType.causal)
        g = DependencyGraph(); g.add_dependency(k.claim_id, f.claim_id)
        out = {x.claim.claim_id: x for x in FailurePropagator().propagate(
            [cr(f, S.inadmissible), cr(k)], g)}
        assert out[k.claim_id].admissibility_state == S.inadmissible
        assert out[k.claim_id].propagated_failure


# ============================================================== extract
class TestExtraction:
    def test_q06_fence_is_longer_than_any_backtick_run(self):
        assert fence_for("plain") == "```"
        assert fence_for("a ``` b") == "````"
        assert fence_for("`````x") == "``````"

    def test_q03_structurally_invalid_claims_are_kept_for_the_guard(self):
        bad = ClaimDraft(claim_type="causal", epistemic_modality="asserted",
                         proposition="Fault caused queue", source_span="s", scope="A", assumptions=[])
        ok = ClaimDraft(claim_type="factual", epistemic_modality="asserted",
                        proposition="Density is 87", source_span="s", scope="A")
        warnings = []
        out = _promote_drafts([ok, bad], warnings)
        assert [c.claim_type for c in out] == [ClaimType.factual, ClaimType.causal]
        assert len(warnings) == 1 and "StructuralGuard" in warnings[0].reason

    def test_field_level_invalid_drafts_are_excluded_with_warning(self):
        bad = ClaimDraft.model_construct(
            claim_id="x", claim_type=ClaimType.factual,
            epistemic_modality=EpistemicModality.asserted, proposition="",
            source_span="s", scope="A", conditions=[], assumptions=[],
            confidence_language=None, time_horizon=None, depends_on=[], negates=[])
        w = []
        assert _promote_drafts([bad], w) == [] and "excluded" in w[0].reason

    def test_source_span_check_ignores_whitespace_reflow(self):
        c = claim(prop="Density is 87 veh/km on Corridor A")
        c = c.model_copy(update={"source_span": "Density is 87 veh/km\n   on Corridor A"})
        kept, warns = _check_source_spans([c], "Density is 87 veh/km on Corridor A.")
        assert kept == [c] and not warns

    def test_fabricated_source_span_is_still_dropped(self):
        c = claim().model_copy(update={"source_span": "text that is not there"})
        kept, warns = _check_source_spans([c], "Density is 87 veh/km.")
        assert kept == [] and len(warns) == 1


# =========================================================== end-to-end
class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_q03_one_invalid_claim_no_longer_aborts_the_run(self):
        bad = dict(claim_type="causal", epistemic_modality="asserted",
                   proposition="The fault caused the queue", source_span="Sensor CAM-042",
                   scope="Corridor A", assumptions=[])
        s = Script(good_claims() + [bad])
        r = await make_validator(s).validate(make_input())
        assert len(r.claim_results) == 4
        assert r.verdict.code == VerdictCode.rejected
        intrinsic = [c for c in r.claim_results
                     if c.admissibility_state == S.inadmissible and not c.propagated_failure]
        assert len(intrinsic) == 1 and intrinsic[0].claim.claim_type == ClaimType.causal
        # the same-scope normative claim inherits the failure (hard propagation)
        assert any(c.propagated_failure for c in r.claim_results)

    @pytest.mark.asyncio
    async def test_q05_pipeline_warnings_returned_without_audit_report(self):
        mix = good_claims(); mix[0] = dict(mix[0], source_span="not in the response")
        r = await make_validator(Script(mix)).validate(make_input())
        assert r.audit_report is None
        assert any("not found verbatim" in w for w in r.pipeline_warnings)

    @pytest.mark.asyncio
    async def test_p16_empty_extraction_escalates_and_audits(self):
        r = await make_validator(Script([]), audit=True).validate(
            make_input(ActionClass.automate, ConsequenceClass.critical))
        assert r.verdict.code == VerdictCode.escalated
        assert r.audit_report is not None
        r2 = await make_validator(Script([]), audit=True).validate(
            make_input(ActionClass.decide, ConsequenceClass.significant))
        assert r2.verdict.code == VerdictCode.escalated

    @pytest.mark.asyncio
    async def test_p17_llm_outage_in_guards_is_visible_and_not_a_violation(self):
        s = Script(good_claims(), fail_guards=True)
        r = await make_validator(s).validate(make_input(ActionClass.recommend, ConsequenceClass.significant))
        states = {c.admissibility_state for c in r.claim_results}
        assert S.contradicted not in states and S.inadmissible not in states
        assert any("could not run" in w for w in r.pipeline_warnings)
        assert r.verdict.code != VerdictCode.accepted

    @pytest.mark.asyncio
    async def test_audit_counts_match_final_results(self):
        r = await make_validator(Script(good_claims()), audit=True).validate(make_input())
        counts = r.audit_report.claims_by_admissibility
        from collections import Counter
        assert counts == {str(k): v for k, v in Counter(c.admissibility_state for c in r.claim_results).items()}
