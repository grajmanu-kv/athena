"""
core.models.result
==================
All output types produced by the Athena validation pipeline.

The result hierarchy:

    ValidationResult            — top-level output of EpistemicValidator.validate()
    ├── list[ClaimResult]       — per-claim verdict with full annotation
    │   ├── Claim               — the extracted claim (from claim.py)
    │   ├── list[GuardResult]   — results from each guard that ran
    │   └── AdmissibilityState  — epistemic admissibility assessment
    ├── Verdict                 — aggregate decision
    └── AuditReport | None      — full audit trail (opt-in)

Design notes
------------
* ClaimResult wraps a Claim immutably. Pipeline stages produce new
  ClaimResult objects — they never mutate Claim instances.

* GuardResult carries both the guard outcome and the guard's identity,
  so the audit trail can explain which guard fired on which claim and why.

* AdmissibilityState is the five-state epistemic classification from the
  spec. It is not a boolean. 'unsubstantiated' ≠ 'contradicted'.

* Verdict carries both a VerdictCode (the decision) and a RiskLevel
  (the derived risk that determined the decision threshold). Consuming
  systems that want to know *why* the verdict was reached look at RiskLevel
  and the per-claim ClaimResults.

* AuditReport is opt-in (ValidationInput.include_audit_report=False by
  default). It includes the full DecomposedPrompt, all intermediate
  pipeline states, and extraction_notes from decomposition.
  It is never None when include_audit_report=True.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, computed_field, SkipValidation

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AdmissibilityState(StrEnum):
    """
    The five-state epistemic admissibility classification for a claim.

    verified        : the claim is consistent with known_facts or injected_facts
                      and passes all structural and guard checks.

    plausible       : the claim is structurally sound and consistent with
                      domain knowledge but cannot be independently verified
                      from available facts. Acceptable for inform/recommend
                      action classes; flagged for decide/automate.

    unsubstantiated : the claim makes an assertion for which no supporting
                      evidence is available in known_facts or injected_facts.
                      Absence of evidence — not evidence of absence.

    contradicted    : the claim conflicts with known_facts, injected_facts,
                      or another claim in the same response.
                      Evidence of absence — stronger than unsubstantiated.

    inadmissible    : the claim violates a structural guard invariant
                      (e.g. predictive claim without time_horizon,
                       causal claim without assumptions) or a declared
                      boundary. Cannot be used regardless of risk level.
    """

    verified        = "verified"
    plausible       = "plausible"
    unsubstantiated = "unsubstantiated"
    contradicted    = "contradicted"
    inadmissible    = "inadmissible"


class RiskLevel(StrEnum):
    """
    Derived risk level for the validation context.
    Computed from ActionClass × ConsequenceClass in risk_matrix.py.
    Not directly settable by consuming systems.

    low      : auto-accept if no inadmissible claims
    medium   : accept with warnings if no inadmissible claims
    high     : human review required; inadmissible claims cause rejection
    critical : always rejected + mandatory human review
    """

    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class VerdictCode(StrEnum):
    """
    The aggregate decision outcome for the validation.

    accepted            : all claims passed guards and admissibility.
    accepted_with_warnings : claims passed guards but some are unsubstantiated
                          or have soft warnings (vague time horizons, implicit
                          constraints, undeclared terms).
    rejected            : one or more claims are inadmissible or contradicted,
                          or risk level is critical.
    escalated           : risk level requires human review; pipeline cannot
                          make a final determination autonomously.
    """

    accepted              = "accepted"
    accepted_with_warnings = "accepted_with_warnings"
    rejected              = "rejected"
    escalated             = "escalated"


class GuardOutcome(StrEnum):
    """
    Result of a single guard's evaluation of a single claim.

    passed  : guard found no violation.
    warning : guard found a soft issue (does not block acceptance).
    failed  : guard found a hard violation (contributes to inadmissibility).
    skipped : guard determined it was not applicable to this claim type.
    errored : guard could not run (an infrastructure fault such as an LLM
              outage). This is ABSENCE of evidence, never evidence of a
              violation: it must not be reported as a failed check.
    """

    passed  = "passed"
    warning = "warning"
    failed  = "failed"
    skipped = "skipped"
    errored = "errored"


# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------


class GuardResult(BaseModel):
    """
    The result of one guard's evaluation of one claim.

    Immutable. Produced by guard implementations; collected by GuardRunner.
    """

    model_config = {"frozen": True}

    guard_id: Annotated[
        str,
        Field(description="Stable identifier for the guard class. e.g. 'StructuralGuard'"),
    ]

    outcome: Annotated[
        GuardOutcome,
        Field(description="Whether the guard passed, warned, failed, or was skipped."),
    ]

    message: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Human-readable explanation of the guard's finding. "
                "Required regardless of outcome — even a pass should "
                "state what was checked and that it passed."
            ),
        ),
    ]

    detail: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description=(
                "Structured key-value evidence supporting the message. "
                "e.g. {'violated_boundary': 'Must not infer causation from correlation', "
                "       'offending_phrase': 'was caused by'}"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# ClaimResult
# ---------------------------------------------------------------------------


class ClaimResult(BaseModel):
    """
    A Claim annotated with the full results of pipeline processing.

    Produced by the assess stage (Step 4) after guards have run (Step 3).
    The dependency resolution stage (Step 5) may update admissibility_state
    via propagation — it produces a new ClaimResult, never mutates this one.
    """

    model_config = {"frozen": True}

    claim: Annotated[
        "Claim",
        # Claim's cross-field invariants are enforced where claims are
        # CONSTRUCTED. Re-running them here would make it impossible to carry a
        # structurally invalid claim to the StructuralGuard (which exists to
        # mark it inadmissible) instead of losing it at extraction.
        SkipValidation,
        Field(description="The extracted claim. Immutable."),
    ]

    guard_results: Annotated[
        list[GuardResult],
        Field(
            default_factory=list,
            description="Results from all guards that evaluated this claim.",
        ),
    ]

    admissibility_state: Annotated[
        AdmissibilityState,
        Field(description="Epistemic admissibility classification for this claim."),
    ]

    admissibility_rationale: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Human-readable explanation of why this admissibility state "
                "was assigned. Required — the audit trail must be able to "
                "explain every decision."
            ),
        ),
    ]

    propagated_failure: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "True when this claim's admissibility_state was downgraded "
                "due to failure propagation from an upstream dependency "
                "in the dependency graph. "
                "Distinguishes intrinsic failure from inherited failure."
            ),
        ),
    ] = False

    warnings: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Soft warnings raised by admissibility checkers (as opposed to "
                "guards), e.g. 'hypothetical at high risk: premise must be "
                "acknowledged'. They count towards has_warnings so the decision "
                "gate sees them."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_admissible(self) -> bool:
        """True when the claim is verified or plausible."""
        return self.admissibility_state in (
            AdmissibilityState.verified,
            AdmissibilityState.plausible,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_warnings(self) -> bool:
        """True when any guard or checker produced a warning."""
        return bool(self.warnings) or any(
            g.outcome in (GuardOutcome.warning, GuardOutcome.errored)
            for g in self.guard_results
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_guards(self) -> list[GuardResult]:
        """Guards that produced a failed outcome for this claim."""
        return [g for g in self.guard_results if g.outcome == GuardOutcome.failed]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    """
    Aggregate decision outcome for the full validation.
    """

    model_config = {"frozen": True}

    code: Annotated[
        VerdictCode,
        Field(description="The decision outcome."),
    ]

    risk_level: Annotated[
        RiskLevel,
        Field(
            description=(
                "Risk level derived from ActionClass × ConsequenceClass. "
                "Determines the threshold applied to reach this verdict."
            ),
        ),
    ]

    rationale: Annotated[
        str,
        Field(
            min_length=1,
            description="Human-readable explanation of why this verdict was reached.",
        ),
    ]

    inadmissible_claim_ids: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="claim_ids of claims that are inadmissible or contradicted.",
        ),
    ]

    warning_claim_ids: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="claim_ids of claims that passed but carry warnings.",
        ),
    ]


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------


class AuditReport(BaseModel):
    """
    Full audit trail for a validation run. Opt-in.

    Includes all intermediate pipeline artifacts and metadata.
    Designed to be serialisable to JSON for storage and review.
    """

    model_config = {"frozen": True}

    run_id: Annotated[
        str,
        Field(description="UUID4 identifier for this validation run."),
    ]

    timestamp: Annotated[
        datetime,
        Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description="UTC timestamp of when validation completed.",
        ),
    ]

    decomposed_prompt: Annotated[
        "DecomposedPrompt",
        Field(description="The structured decomposition produced by Step 0."),
    ]

    total_claims_extracted: Annotated[
        int,
        Field(ge=0, description="Number of claims extracted in Step 1."),
    ]

    claims_by_type: Annotated[
        dict[str, int],
        Field(
            default_factory=dict,
            description=(
                "Count of claims by ClaimType. "
                "e.g. {'factual': 3, 'causal': 1, 'predictive': 2, 'normative': 1}"
            ),
        ),
    ]

    claims_by_admissibility: Annotated[
        dict[str, int],
        Field(
            default_factory=dict,
            description=(
                "Count of claims by AdmissibilityState. "
                "e.g. {'verified': 2, 'plausible': 2, 'unsubstantiated': 1, "
                "       'contradicted': 0, 'inadmissible': 1}"
            ),
        ),
    ]

    propagated_failure_count: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Number of claims whose admissibility was downgraded "
                "due to dependency graph failure propagation."
            ),
        ),
    ] = 0

    extraction_notes: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Verbatim extraction_notes from the decomposition agent, "
                "if any were produced."
            ),
        ),
    ] = None

    pipeline_warnings: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Non-fatal pipeline-level warnings. "
                "e.g. 'No constraints declared — constraint guard had nothing to check', "
                "     'Undeclared technical terms found: [anomaly, density]'"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# ValidationResult — top-level output
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """
    The complete output of EpistemicValidator.validate().

    This is the single object a consuming system receives.
    It contains everything needed to act on the validation outcome.
    """

    model_config = {"frozen": True}

    claim_results: Annotated[
        list["ClaimResult"],
        Field(
            default_factory=list,
            description=(
                "Per-claim results in extraction order. "
                "Each carries the Claim, guard results, admissibility state, "
                "and rationale."
            ),
        ),
    ]

    verdict: Annotated[
        Verdict,
        Field(description="Aggregate decision outcome."),
    ]

    audit_report: Annotated[
        AuditReport | None,
        Field(
            default=None,
            description=(
                "Full audit trail. None when include_audit_report=False "
                "(the default). Always populated when include_audit_report=True."
            ),
        ),
    ] = None

    pipeline_warnings: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Non-fatal pipeline-level warnings, ALWAYS returned (also when "
                "include_audit_report=False): claims dropped at extraction, "
                "guards that could not run, empty extraction. A claim that "
                "disappears from the result must never do so silently."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """True when verdict is accepted or accepted_with_warnings."""
        return self.verdict.code in (
            VerdictCode.accepted,
            VerdictCode.accepted_with_warnings,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inadmissible_claims(self) -> list["ClaimResult"]:
        """Claims that are inadmissible or contradicted."""
        return [
            cr for cr in self.claim_results
            if cr.admissibility_state in (
                AdmissibilityState.inadmissible,
                AdmissibilityState.contradicted,
            )
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def admissible_claims(self) -> list["ClaimResult"]:
        """Claims that are verified or plausible."""
        return [cr for cr in self.claim_results if cr.is_admissible]