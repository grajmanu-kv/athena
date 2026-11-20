"""
util.logging
============
Structured audit logging helpers for the Athena pipeline.

Emits structured loguru log entries at pipeline stage boundaries.
All entries include a run_id for correlation across a single validation run.

Consuming systems that want to ship audit logs to external systems
(OpenTelemetry, structlog, ELK) should configure loguru sinks on their
side — Athena does not prescribe a log destination.

Log levels used
---------------
DEBUG   — per-claim guard/assessment details (high volume)
INFO    — stage entry/exit, per-run summary
WARNING — soft guard warnings, missing optional inputs
ERROR   — guard/checker exceptions, extraction failures
"""

from __future__ import annotations

from loguru import logger

from athena.core.models.result import ClaimResult, ValidationResult, Verdict


def log_pipeline_start(run_id: str, domain: str, action_class: str) -> None:
    logger.info(
        "[athena] run={} | START | domain={} | action_class={}",
        run_id, domain, action_class,
    )


def log_decompose_complete(run_id: str, implicit_count: int) -> None:
    logger.info(
        "[athena] run={} | DECOMPOSE complete | "
        "implicit_constraints={} extracted",
        run_id, implicit_count,
    )


def log_extract_complete(
    run_id: str,
    claim_count: int,
    warning_count: int,
) -> None:
    logger.info(
        "[athena] run={} | EXTRACT complete | "
        "claims={} | extraction_warnings={}",
        run_id, claim_count, warning_count,
    )


def log_classify_complete(run_id: str, issue_count: int) -> None:
    logger.info(
        "[athena] run={} | CLASSIFY complete | classification_issues={}",
        run_id, issue_count,
    )


def log_guard_complete(
    run_id: str,
    failed_count: int,
    warning_count: int,
) -> None:
    logger.info(
        "[athena] run={} | GUARD complete | "
        "guard_failures={} | guard_warnings={}",
        run_id, failed_count, warning_count,
    )


def log_assess_complete(
    run_id: str,
    state_counts: dict[str, int],
) -> None:
    logger.info(
        "[athena] run={} | ASSESS complete | states={}",
        run_id, state_counts,
    )


def log_resolve_complete(
    run_id: str,
    propagated_count: int,
    negation_pairs: int,
) -> None:
    logger.info(
        "[athena] run={} | RESOLVE complete | "
        "propagated_failures={} | negation_pairs={}",
        run_id, propagated_count, negation_pairs,
    )


def log_verdict(run_id: str, verdict: Verdict) -> None:
    logger.info(
        "[athena] run={} | VERDICT={} | risk={} | "
        "inadmissible={} | warnings={}",
        run_id,
        verdict.code,
        verdict.risk_level,
        len(verdict.inadmissible_claim_ids),
        len(verdict.warning_claim_ids),
    )


def log_claim_detail(run_id: str, cr: ClaimResult) -> None:
    """DEBUG-level per-claim detail."""
    logger.debug(
        "[athena] run={} | claim={} | type={} | modality={} | "
        "admissibility={} | propagated={}",
        run_id,
        cr.claim.claim_id,
        cr.claim.claim_type,
        cr.claim.epistemic_modality,
        cr.admissibility_state,
        cr.propagated_failure,
    )


def log_validation_summary(run_id: str, result: ValidationResult) -> None:
    """INFO-level end-of-run summary."""
    logger.info(
        "[athena] run={} | COMPLETE | verdict={} | "
        "claims={} | admissible={} | inadmissible={}",
        run_id,
        result.verdict.code,
        len(result.claim_results),
        len(result.admissible_claims),
        len(result.inadmissible_claims),
    )