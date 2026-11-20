"""
stages.decide.gate
==================
Step 6 — Decision Gate.

Maps assessed ClaimResults + RiskLevel to a Verdict.

Decision logic
--------------
The gate applies a risk-calibrated threshold:

    critical risk:
        Always VerdictCode.escalated regardless of claim admissibility.
        The pipeline cannot make a final determination — human review
        is mandatory. ConsequenceClass.critical + ActionClass.automate
        is the paradigmatic case.

    high risk:
        Any inadmissible or contradicted claim → VerdictCode.rejected.
        Any unsubstantiated claim → VerdictCode.escalated (human review).
        All plausible/verified → VerdictCode.accepted_with_warnings
        if warnings exist, else VerdictCode.accepted.

    medium risk:
        Any inadmissible or contradicted claim → VerdictCode.rejected.
        Unsubstantiated claims produce warnings but not rejection.
        All plausible/verified → VerdictCode.accepted_with_warnings
        if warnings or unsubstantiated claims exist.

    low risk:
        Any inadmissible or contradicted claim → VerdictCode.rejected.
        (A contradicted claim is a constraint/contract breach or a proven
        conflict; risk level only governs how much *uncertainty* is
        tolerated, not whether a demonstrated violation is.)
        Unsubstantiated claims produce warnings only.
        All plausible/verified → VerdictCode.accepted or
        VerdictCode.accepted_with_warnings.

    no claims at all:
        Nothing was validated, which is not the same as nothing being wrong.
        critical → escalated; high → escalated; medium/low →
        accepted_with_warnings.

Aggregate vs per-claim
-----------------------
The gate operates on the aggregate ClaimResult list. A single
inadmissible claim in a critical risk context rejects the entire
response — not just that claim. Consuming systems inspect
Verdict.inadmissible_claim_ids and Verdict.warning_claim_ids to
understand which claims drove the verdict.

When unsure, block (Invariant #7)
---------------------------------
When the evidence is mixed — e.g. plausible and unsubstantiated claims at
high risk — the gate escalates or rejects rather than accepts. In plain words:
letting a bad claim through (a missed detection) is judged more costly than
holding back a good one (a false alarm) in decision-critical contexts.
"""

from __future__ import annotations

from athena.core.models.input import ActionClass, ConsequenceClass
from athena.core.models.result import (
    AdmissibilityState,
    ClaimResult,
    RiskLevel,
    Verdict,
    VerdictCode,
)
from athena.core.risk.risk_level import derive_risk_level


class DecisionGate:
    """
    Maps ClaimResults + risk context to a Verdict.
    Stateless — no constructor arguments.
    """

    def decide(
        self,
        claim_results: list[ClaimResult],
        action_class: ActionClass,
        consequence_class: ConsequenceClass,
    ) -> Verdict:
        """
        Produce a Verdict from assessed claim results.

        Parameters
        ----------
        claim_results:
            All ClaimResults after guard, assessment, and propagation.
        action_class:
            From IntentContext — determines risk alongside consequence.
        consequence_class:
            From IntentContext — determines risk alongside action.

        Returns
        -------
        Verdict
            The aggregate decision outcome.
        """
        risk_level = derive_risk_level(action_class, consequence_class)

        inadmissible_ids = [
            cr.claim.claim_id for cr in claim_results
            if cr.admissibility_state in (
                AdmissibilityState.inadmissible,
                AdmissibilityState.contradicted,
            )
        ]
        unsubstantiated_ids = [
            cr.claim.claim_id for cr in claim_results
            if cr.admissibility_state == AdmissibilityState.unsubstantiated
        ]
        warning_ids = [
            cr.claim.claim_id for cr in claim_results
            if cr.has_warnings and cr.claim.claim_id not in inadmissible_ids
        ]

        # No claims: nothing was validated. Never a clean 'accepted'.
        if not claim_results and risk_level not in (RiskLevel.critical,):
            empty_note = (
                "No claims were extracted, so nothing was validated. "
            )
            if risk_level == RiskLevel.high:
                return Verdict(
                    code=VerdictCode.escalated,
                    risk_level=risk_level,
                    rationale=empty_note + (
                        "High risk context: an unvalidated response cannot "
                        "be accepted. Human review required."
                    ),
                    inadmissible_claim_ids=[],
                    warning_claim_ids=[],
                )
            return Verdict(
                code=VerdictCode.accepted_with_warnings,
                risk_level=risk_level,
                rationale=empty_note + (
                    f"{risk_level.value.capitalize()} risk context: accepted "
                    f"with warnings."
                ),
                inadmissible_claim_ids=[],
                warning_claim_ids=[],
            )

        # critical: always escalate
        if risk_level == RiskLevel.critical:
            return Verdict(
                code=VerdictCode.escalated,
                risk_level=risk_level,
                rationale=(
                    "Risk level is critical. Automated validation cannot "
                    "make a final determination. Mandatory human review required. "
                    f"({len(inadmissible_ids)} inadmissible/contradicted claim(s), "
                    f"{len(unsubstantiated_ids)} unsubstantiated claim(s).)"
                ),
                inadmissible_claim_ids=inadmissible_ids,
                warning_claim_ids=warning_ids + unsubstantiated_ids,
            )

        # high: reject on inadmissible, escalate on unsubstantiated
        if risk_level == RiskLevel.high:
            if inadmissible_ids:
                return Verdict(
                    code=VerdictCode.rejected,
                    risk_level=risk_level,
                    rationale=(
                        f"High risk context: {len(inadmissible_ids)} claim(s) "
                        f"are inadmissible or contradicted. Rejected."
                    ),
                    inadmissible_claim_ids=inadmissible_ids,
                    warning_claim_ids=warning_ids,
                )
            if unsubstantiated_ids:
                return Verdict(
                    code=VerdictCode.escalated,
                    risk_level=risk_level,
                    rationale=(
                        f"High risk context: {len(unsubstantiated_ids)} claim(s) "
                        f"are unsubstantiated. Epistemic standard too low for "
                        f"high-stakes decision. Human review required."
                    ),
                    inadmissible_claim_ids=[],
                    warning_claim_ids=warning_ids + unsubstantiated_ids,
                )
            if warning_ids:
                return Verdict(
                    code=VerdictCode.accepted_with_warnings,
                    risk_level=risk_level,
                    rationale=(
                        f"High risk context: all claims admissible but "
                        f"{len(warning_ids)} carry soft warnings. "
                        f"Accepted with warnings — review flagged claims."
                    ),
                    inadmissible_claim_ids=[],
                    warning_claim_ids=warning_ids,
                )
            return Verdict(
                code=VerdictCode.accepted,
                risk_level=risk_level,
                rationale=(
                    "High risk context: all claims verified or plausible "
                    "with no warnings. Accepted."
                ),
                inadmissible_claim_ids=[],
                warning_claim_ids=[],
            )

        # medium: reject on inadmissible, warn on unsubstantiated
        if risk_level == RiskLevel.medium:
            if inadmissible_ids:
                return Verdict(
                    code=VerdictCode.rejected,
                    risk_level=risk_level,
                    rationale=(
                        f"Medium risk context: {len(inadmissible_ids)} claim(s) "
                        f"are inadmissible or contradicted. Rejected."
                    ),
                    inadmissible_claim_ids=inadmissible_ids,
                    warning_claim_ids=warning_ids,
                )
            all_warning_ids = list(
                dict.fromkeys(warning_ids + unsubstantiated_ids)
            )
            if all_warning_ids:
                return Verdict(
                    code=VerdictCode.accepted_with_warnings,
                    risk_level=risk_level,
                    rationale=(
                        f"Medium risk context: no inadmissible claims but "
                        f"{len(unsubstantiated_ids)} unsubstantiated and "
                        f"{len(warning_ids)} with warnings. "
                        f"Accepted with warnings."
                    ),
                    inadmissible_claim_ids=[],
                    warning_claim_ids=all_warning_ids,
                )
            return Verdict(
                code=VerdictCode.accepted,
                risk_level=risk_level,
                rationale=(
                    "Medium risk context: all claims verified or plausible. "
                    "Accepted."
                ),
                inadmissible_claim_ids=[],
                warning_claim_ids=[],
            )

        # low: reject only on inadmissible
        if inadmissible_ids:
            return Verdict(
                code=VerdictCode.rejected,
                risk_level=risk_level,
                rationale=(
                    f"Low risk context: {len(inadmissible_ids)} claim(s) "
                    f"are inadmissible or contradicted. Rejected."
                ),
                inadmissible_claim_ids=inadmissible_ids,
                warning_claim_ids=warning_ids,
            )
        all_warning_ids = list(
            dict.fromkeys(warning_ids + unsubstantiated_ids)
        )
        if all_warning_ids:
            return Verdict(
                code=VerdictCode.accepted_with_warnings,
                risk_level=risk_level,
                rationale=(
                    f"Low risk context: no inadmissible claims. "
                    f"{len(all_warning_ids)} soft issue(s). "
                    f"Accepted with warnings."
                ),
                inadmissible_claim_ids=[],
                warning_claim_ids=all_warning_ids,
            )
        return Verdict(
            code=VerdictCode.accepted,
            risk_level=risk_level,
            rationale="Low risk context: all claims admissible. Accepted.",
            inadmissible_claim_ids=[],
            warning_claim_ids=[],
        )