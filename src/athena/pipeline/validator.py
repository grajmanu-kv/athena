"""
pipeline.validator
==================
EpistemicValidator — the single public entry point for the Athena library.

Orchestrates all six pipeline stages sequentially:

    Stage 0: DecomposeAgent      → DecomposedPrompt
    Stage 1: ExtractAgent        → list[Claim]
    Stage 2: ClaimClassifier     → list[ClassificationResult]
    Stage 3: GuardRunner         → dict[claim_id, list[GuardResult]]
    Stage 4: AdmissibilityAssessor → list[ClaimResult]
    Stage 5: DependencyBuilder   → DependencyGraph
             FailurePropagator   → list[ClaimResult] (resolved)
    Stage 6: DecisionGate        → Verdict

Public API
----------
    validator = EpistemicValidator(
        model=my_model,
        guards=[MyDomainGuard()],
        admissibility_checkers=[MyFactChecker()],
        dependency_builder=None,      # uses HeuristicDependencyBuilder
        include_audit_report=False,
    )

    result: ValidationResult = await validator.validate(input)

All constructor parameters except `model` are optional.
"""

from __future__ import annotations

from collections import Counter

from pydantic_ai.models import Model

from athena.core.exceptions import AthenaConfigurationError
from athena.core.models.input import ValidationInput
from athena.core.models.result import (
    AuditReport,
    ValidationResult,
)
from athena.core.protocols.admissibility import AdmissibilityCheckerProtocol
from athena.core.protocols.dependency import DependencyGraphBuilderProtocol
from athena.core.protocols.guard import GuardProtocol
from athena.pipeline.context import PipelineContext
from athena.stages.assess.assessor import AdmissibilityAssessor
from athena.stages.classifier.agent import ClaimClassifier
from athena.stages.decide.gate import DecisionGate
from athena.stages.decompose.agent import DecomposeAgent
from athena.stages.extract.agent import ExtractAgent
from athena.stages.guard.builtin.boundary import BoundaryGuard
from athena.stages.guard.builtin.constraint import ConstraintGuard
from athena.stages.guard.builtin.structural import StructuralGuard
from athena.stages.guard.builtin.terminology import TerminologyGuard
from athena.stages.guard.runner import GuardRunner
from athena.stages.resolve.builder import HeuristicDependencyBuilder
from athena.stages.resolve.propagator import FailurePropagator
from athena.util.ids import new_id
from athena.util.logging import (
    log_assess_complete,
    log_claim_detail,
    log_classify_complete,
    log_decompose_complete,
    log_extract_complete,
    log_guard_complete,
    log_pipeline_start,
    log_resolve_complete,
    log_validation_summary,
    log_verdict,
)


class EpistemicValidator:
    """
    Orchestrates the full Athena epistemic validation pipeline.

    Parameters
    ----------
    model:
        PydanticAI Model instance. Used by DecomposeAgent, ExtractAgent,
        and all LLM-backed guards (ConstraintGuard, BoundaryGuard,
        TerminologyGuard). The calling system provides this — Athena
        is model-agnostic.
    guards:
        Additional GuardProtocol implementations registered by the
        consuming system. Run after all built-in guards.
    admissibility_checkers:
        AdmissibilityCheckerProtocol implementations registered by the
        consuming system. Run after built-in checkers.
    dependency_builder:
        Override for the default HeuristicDependencyBuilder.
        Must implement DependencyGraphBuilderProtocol.
    include_audit_report:
        When True, ValidationResult.audit_report is populated with
        the full pipeline audit trail. Default: False.
    """

    def __init__(
        self,
        model: Model,
        guards: list[GuardProtocol] | None = None,
        admissibility_checkers: list[AdmissibilityCheckerProtocol] | None = None,
        dependency_builder: DependencyGraphBuilderProtocol | None = None,
        include_audit_report: bool = False,
    ) -> None:
        if model is None:
            raise AthenaConfigurationError(
                "EpistemicValidator requires a pydantic_ai Model instance. "
                "Provide a model via the 'model' parameter."
            )

        self._model = model
        self._include_audit_report = include_audit_report

        # Stage agents
        self._decompose_agent = DecomposeAgent(model=model)
        self._extract_agent = ExtractAgent(model=model)

        # Built-in guards (always run, in declared order)
        builtin_guards: list[GuardProtocol] = [
            StructuralGuard(),
            ConstraintGuard(model=model),
            BoundaryGuard(model=model),
            TerminologyGuard(model=model),
        ]

        # Validate consumer guard IDs are unique
        consumer_guards = guards or []
        all_guard_ids = [g.guard_id for g in builtin_guards + consumer_guards]
        seen: set[str] = set()
        for gid in all_guard_ids:
            if gid in seen:
                raise AthenaConfigurationError(
                    f"Duplicate guard_id '{gid}'. "
                    f"All registered guards must have unique IDs."
                )
            seen.add(gid)

        self._guard_runner = GuardRunner(
            guards=builtin_guards + consumer_guards
        )

        # Assessor with consumer checkers
        self._assessor = AdmissibilityAssessor(
            consumer_checkers=admissibility_checkers or []
        )

        # Dependency builder
        self._dependency_builder: DependencyGraphBuilderProtocol = (
            dependency_builder or HeuristicDependencyBuilder()
        )

        self._propagator = FailurePropagator()
        self._classifier = ClaimClassifier()
        self._gate = DecisionGate()

    async def validate(self, input: ValidationInput) -> ValidationResult:
        """
        Run the full epistemic validation pipeline.

        Parameters
        ----------
        input:
            The complete validation input (prompt, intent, domain, response).

        Returns
        -------
        ValidationResult
            The structured verdict with per-claim results and optional
            audit report.

        Raises
        ------
        DecompositionFailure
            If Step 0 fails after retries.
        ClaimExtractionFailure
            If Step 1 fails after retries.
        """
        run_id = new_id()
        ctx = PipelineContext(
            validation_input=input,
            run_id=run_id,
        )

        log_pipeline_start(
            run_id,
            domain=input.intent_context.domain,
            action_class=input.intent_context.action_class,
        )

        # ------------------------------------------------------------------
        # Stage 0: Prompt decomposition
        # ------------------------------------------------------------------
        ctx.decomposed_prompt = await self._decompose_agent.run(
            prompt=input.prompt,
            domain_info=input.domain_info,
        )

        # Store the LLM response on DecomposedPrompt so guards
        # can evaluate response-level constraints (e.g. citation checks)
        # without requiring a GuardProtocol signature change.
        ctx.decomposed_prompt = ctx.decomposed_prompt.model_copy(
            update={"llm_response": input.llm_response}
        )

        if not ctx.decomposed_prompt.has_any_contract:
            ctx.add_warning(
                "No constraints or boundaries available after decomposition. "
                "ConstraintGuard and BoundaryGuard will skip all claims."
            )

        log_decompose_complete(
            run_id,
            implicit_count=(
                len(ctx.decomposed_prompt.implicit_constraints)
                + len(ctx.decomposed_prompt.implicit_boundaries)
            ),
        )

        # ------------------------------------------------------------------
        # Stage 1: Claim extraction
        # ------------------------------------------------------------------
        ctx.extracted_claims, ctx.extraction_warnings = (
            await self._extract_agent.run(
                llm_response=input.llm_response,
                decomposed_prompt=ctx.decomposed_prompt,
                intent_context=input.intent_context,
            )
        )

        for w in ctx.extraction_warnings:
            ctx.add_warning(w.reason)

        log_extract_complete(
            run_id,
            claim_count=len(ctx.extracted_claims),
            warning_count=len(ctx.extraction_warnings),
        )

        if not ctx.extracted_claims:
            ctx.add_warning(
                "No claims extracted from the LLM response. "
                "The response may contain no verifiable propositions."
            )
            return self._build_empty_result(ctx)

        # Guards that could not run are surfaced, never silent.
        # (collected after stage 3, below)

        # ------------------------------------------------------------------
        # Stage 2: Classification
        # ------------------------------------------------------------------
        ctx.classification_results = self._classifier.classify_all(
            ctx.extracted_claims
        )
        issue_count = sum(
            1 for cr in ctx.classification_results if cr.has_issues
        )
        for cr in ctx.classification_results:
            for warning in cr.warnings:
                ctx.add_warning(
                    f"[classify:{cr.claim_id[:8]}] {warning}"
                )
        log_classify_complete(run_id, issue_count=issue_count)

        # ------------------------------------------------------------------
        # Stage 3: Guard execution
        # ------------------------------------------------------------------
        guard_records = await self._guard_runner.run_all(
            claims=ctx.extracted_claims,
            decomposed_prompt=ctx.decomposed_prompt,
            intent_context=input.intent_context,
        )
        ctx.guard_results_map = {
            record.claim_id: record.guard_results
            for record in guard_records
        }
        failed = sum(1 for r in guard_records if r.has_failures)
        warned = sum(1 for r in guard_records if r.has_warnings)
        errored_ids = sorted({
            gr.guard_id
            for rec in guard_records
            for gr in rec.guard_results
            if gr.outcome.value == "errored"
        })
        if errored_ids:
            ctx.add_warning(
                "Guard(s) could not run for at least one claim (model/guard "
                f"error): {', '.join(errored_ids)}. Affected claims are "
                "unverified, not violating."
            )
        log_guard_complete(run_id, failed_count=failed, warning_count=warned)

        # ------------------------------------------------------------------
        # Stage 4: Admissibility assessment
        # ------------------------------------------------------------------
        ctx.assessed_claim_results = await self._assessor.assess_all(
            claims=ctx.extracted_claims,
            guard_results_map=ctx.guard_results_map,
            decomposed_prompt=ctx.decomposed_prompt,
            intent_context=input.intent_context,
        )
        log_assess_complete(
            run_id,
            state_counts={
                k: v for k, v in Counter(
                    cr.admissibility_state for cr in ctx.assessed_claim_results
                ).items()
            },
        )
        for cr in ctx.assessed_claim_results:
            log_claim_detail(run_id, cr)

        # ------------------------------------------------------------------
        # Stage 5: Dependency graph + failure propagation
        # ------------------------------------------------------------------
        ctx.dependency_graph = await self._dependency_builder.build(
            claim_results=ctx.assessed_claim_results,
            decomposed_prompt=ctx.decomposed_prompt,
            intent_context=input.intent_context,
        )
        ctx.resolved_claim_results = self._propagator.propagate(
            claim_results=ctx.assessed_claim_results,
            graph=ctx.dependency_graph,
        )
        propagated = sum(
            1 for cr in ctx.resolved_claim_results if cr.propagated_failure
        )
        negation_pairs = sum(
            len(ctx.dependency_graph.negations_of(cr.claim.claim_id))
            for cr in ctx.resolved_claim_results
        ) // 2
        log_resolve_complete(
            run_id,
            propagated_count=propagated,
            negation_pairs=negation_pairs,
        )

        # ------------------------------------------------------------------
        # Stage 6: Decision gate
        # ------------------------------------------------------------------
        verdict = self._gate.decide(
            claim_results=ctx.final_claim_results,
            action_class=input.intent_context.action_class,
            consequence_class=input.intent_context.consequence_class,
        )
        log_verdict(run_id, verdict)

        # ------------------------------------------------------------------
        # Assemble result
        # ------------------------------------------------------------------
        # Report the FINAL (post-propagation) states, matching claim_results.
        state_counts = Counter(
            cr.admissibility_state for cr in ctx.final_claim_results
        )
        audit_report = None
        if self._include_audit_report:
            audit_report = self._build_audit_report(ctx, state_counts)

        result = ValidationResult(
            claim_results=ctx.final_claim_results,
            verdict=verdict,
            audit_report=audit_report,
            pipeline_warnings=list(ctx.pipeline_warnings),
        )
        log_validation_summary(run_id, result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_empty_result(self, ctx: PipelineContext) -> ValidationResult:
        """
        Build a ValidationResult when no claims were extracted.

        The verdict comes from the DecisionGate so it is risk-calibrated:
        an empty extraction can never yield a clean accept, and at high or
        critical risk it escalates. The audit report is populated when
        requested, exactly as for a normal run.
        """
        intent = ctx.validation_input.intent_context
        verdict = self._gate.decide(
            claim_results=[],
            action_class=intent.action_class,
            consequence_class=intent.consequence_class,
        )
        audit_report = (
            self._build_audit_report(ctx, Counter())
            if self._include_audit_report
            else None
        )
        return ValidationResult(
            claim_results=[],
            verdict=verdict,
            audit_report=audit_report,
            pipeline_warnings=list(ctx.pipeline_warnings),
        )

    def _build_audit_report(
        self,
        ctx: PipelineContext,
        state_counts: Counter,
    ) -> AuditReport:
        """Assemble the full AuditReport from pipeline context."""
        assert ctx.decomposed_prompt is not None

        claims_by_type: dict[str, int] = {}
        for claim in ctx.extracted_claims:
            ct = str(claim.claim_type)
            claims_by_type[ct] = claims_by_type.get(ct, 0) + 1

        return AuditReport(
            run_id=ctx.run_id,
            decomposed_prompt=ctx.decomposed_prompt,
            total_claims_extracted=len(ctx.extracted_claims),
            claims_by_type=claims_by_type,
            claims_by_admissibility={
                str(k): v for k, v in state_counts.items()
            },
            propagated_failure_count=sum(
                1 for cr in ctx.resolved_claim_results
                if cr.propagated_failure
            ),
            extraction_notes=ctx.decomposed_prompt.extraction_notes,
            pipeline_warnings=list(ctx.pipeline_warnings),
        )