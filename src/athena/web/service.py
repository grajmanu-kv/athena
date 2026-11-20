"""
athena.web.service
==================
Turns an HTTP request into a ValidationInput, runs the real Athena pipeline,
and flattens the ValidationResult into a JSON-safe payload for the UI.

The verdict, per-claim states and guard outcomes are the library's own output,
passed through unchanged. This module adds only (a) counts and (b) honest
"evidence notes" about what the run could and could not establish.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from pydantic_ai.models import Model

from athena import (
    DomainInfo,
    EpistemicValidator,
    IntentContext,
    ValidationInput,
)
from athena.core.models.result import ClaimResult, GuardOutcome, ValidationResult
from athena.core.risk.risk_level import derive_risk_level
from athena.web.known_facts import KnownFactsChecker
from athena.web.schemas import ValidateRequest

STATE_ORDER = ["verified", "plausible", "unsubstantiated", "contradicted", "inadmissible"]


def build_input(req: ValidateRequest) -> ValidationInput:
    return ValidationInput(
        prompt=req.query,
        llm_response=req.response,
        intent_context=IntentContext(
            goal=req.goal,
            actor=req.actor,
            action_class=req.action_class,
            consequence_class=req.consequence_class,
            decision_window=req.decision_window,
            domain=req.domain,
        ),
        domain_info=DomainInfo(
            description=req.description or f"General {req.domain} context.",
            term_definitions=req.term_definitions,
            known_facts=req.known_facts,
            constraints=req.constraints,
            boundaries=req.boundaries,
        ),
    )


async def run_validation(req: ValidateRequest, model: Model, timeout_s: float) -> dict[str, Any]:
    vin = build_input(req)
    validator = EpistemicValidator(
        model=model,
        admissibility_checkers=[KnownFactsChecker(model, req.known_facts)],
        include_audit_report=True,
    )
    async with asyncio.timeout(timeout_s):
        result = await validator.validate(vin)
    return serialise(result, req)


# ---------------------------------------------------------------------------


def _claim_payload(cr: ClaimResult, text_index: dict[str, str]) -> dict[str, Any]:
    c = cr.claim
    th = c.time_horizon
    return {
        "claim_id": c.claim_id,
        "type": str(c.claim_type),
        "modality": str(c.epistemic_modality),
        "proposition": c.proposition,
        "source_span": c.source_span,
        "scope": c.scope,
        "assumptions": list(c.assumptions),
        "conditions": list(c.conditions),
        "confidence_language": c.confidence_language,
        "time_horizon": {"raw": th.raw, "normalized": th.normalized} if th else None,
        "depends_on": sorted(c.depends_on),
        "negates": sorted(c.negates),
        "state": str(cr.admissibility_state),
        "rationale": cr.admissibility_rationale,
        "propagated_failure": cr.propagated_failure,
        "warnings": list(cr.warnings),
        "has_warnings": cr.has_warnings,
        "guards": [
            {
                "guard_id": g.guard_id,
                "outcome": str(g.outcome),
                "message": g.message,
                "detail": {k: str(v) for k, v in (g.detail or {}).items()},
            }
            for g in cr.guard_results
        ],
    }


def serialise(result: ValidationResult, req: ValidateRequest) -> dict[str, Any]:
    v = result.verdict
    claims = [_claim_payload(cr, {}) for cr in result.claim_results]
    counts = Counter(c["state"] for c in claims)

    audit = result.audit_report
    dp = audit.decomposed_prompt if audit else None

    errored = [
        (c["claim_id"], g["guard_id"])
        for c in claims
        for g in c["guards"]
        if g["outcome"] == GuardOutcome.errored.value
    ]
    skipped_guards = sorted(
        {g["guard_id"] for c in claims for g in c["guards"] if g["outcome"] == "skipped"}
    )

    notes: list[dict[str, str]] = []
    if not req.known_facts:
        notes.append({
            "level": "info",
            "text": "No known_facts supplied: nothing can be marked 'verified'. "
                    "The best attainable state is 'plausible'.",
        })
    if errored:
        notes.append({
            "level": "warn",
            "text": f"{len(errored)} guard evaluation(s) could not run (model error). "
                    "Those are treated as missing evidence, not as passes or violations.",
        })
    if not (req.constraints or req.boundaries) and dp is not None and not dp.has_any_contract:
        notes.append({
            "level": "info",
            "text": "No constraints or boundaries were declared or inferred, so the "
                    "constraint and boundary guards had nothing to check.",
        })
    if not claims:
        notes.append({
            "level": "warn",
            "text": "No claims were extracted. The response may hold no verifiable propositions.",
        })

    return {
        "verdict": {
            "code": str(v.code),
            "risk_level": str(v.risk_level),
            "rationale": v.rationale,
            "inadmissible_claim_ids": list(v.inadmissible_claim_ids),
            "warning_claim_ids": list(v.warning_claim_ids),
        },
        "risk_level_derived": str(derive_risk_level(req.action_class, req.consequence_class)),
        "counts": {s: counts.get(s, 0) for s in STATE_ORDER},
        "total_claims": len(claims),
        "propagated_failures": sum(1 for c in claims if c["propagated_failure"]),
        "claims": claims,
        "pipeline_warnings": list(result.pipeline_warnings),
        "notes": notes,
        "skipped_guards": skipped_guards,
        "decomposition": None if dp is None else {
            "explicit_constraints": list(dp.explicit_constraints or []),
            "implicit_constraints": list(dp.implicit_constraints or []),
            "explicit_boundaries": list(dp.explicit_boundaries or []),
            "implicit_boundaries": list(dp.implicit_boundaries or []),
            "term_definitions": dict(dp.term_definitions or {}),
            "undeclared_terms": list(dp.undeclared_terms or []),
            "injected_facts": list(dp.injected_facts or []),
            "extraction_notes": list(dp.extraction_notes or []),
        },
        "run_id": audit.run_id if audit else None,
    }
