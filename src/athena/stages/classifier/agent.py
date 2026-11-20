"""
stages.classify.classifier
==========================
Step 2 — Epistemic Classification.

Deterministic, no LLM call. Validates the internal consistency of each
Claim's claim_type × epistemic_modality combination and enforces the
hard structural preconditions that Pydantic model validators could not
catch at extraction time (because they operate per-field, not cross-field
with pipeline context).

What this stage does
--------------------
1. Validates claim_type × epistemic_modality matrix consistency.
   Some combinations are category errors — they indicate the extraction
   agent misclassified the claim.

2. Validates confidence_language consistency with epistemic_modality.
   An `asserted` claim with hedging language in confidence_language is
   internally inconsistent — one of them is wrong.

3. Validates structural preconditions by claim_type:
   - causal:    assumptions non-empty  (already enforced by Pydantic, but
                re-checked here with pipeline context for audit trail)
   - predictive: time_horizon present  (same)

4. Produces a ClassificationResult per claim — a lightweight annotation
   carrying any consistency warnings or reclassification suggestions.
   It does NOT mutate the Claim.

What this stage does NOT do
---------------------------
- It does not assess admissibility (Step 4).
- It does not check against constraints or boundaries (Step 3).
- It does not call an LLM.
- It does not reject claims — it annotates them.
  A claim with a classification warning is still processed by guards
  and admissibility checkers. The warning surfaces in the audit trail.

Design: ClassificationResult is not a Pydantic model — it is a simple
dataclass produced and consumed within the pipeline. It is not serialised
directly; its content is absorbed into AuditReport.pipeline_warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from athena.core.models.claim import Claim, ClaimType, EpistemicModality


# ---------------------------------------------------------------------------
# Consistency matrix
# ---------------------------------------------------------------------------

# Maps (ClaimType, EpistemicModality) → True if valid, False if category error
# A "category error" means the combination is internally contradictory
# in a way that indicates misclassification, not just an unusual claim.
_VALIDITY_MATRIX: dict[tuple[ClaimType, EpistemicModality], bool] = {
    # factual claims — all modalities valid
    (ClaimType.factual, EpistemicModality.asserted):     True,
    (ClaimType.factual, EpistemicModality.hedged):       True,
    (ClaimType.factual, EpistemicModality.speculative):  True,
    (ClaimType.factual, EpistemicModality.hypothetical): True,

    # causal claims — all modalities valid
    # (speculative causal = "this might have caused that" — valid but weak)
    (ClaimType.causal, EpistemicModality.asserted):      True,
    (ClaimType.causal, EpistemicModality.hedged):        True,
    (ClaimType.causal, EpistemicModality.speculative):   True,
    (ClaimType.causal, EpistemicModality.hypothetical):  True,

    # predictive claims — all modalities valid
    (ClaimType.predictive, EpistemicModality.asserted):     True,
    (ClaimType.predictive, EpistemicModality.hedged):       True,
    (ClaimType.predictive, EpistemicModality.speculative):  True,
    (ClaimType.predictive, EpistemicModality.hypothetical): True,

    # normative claims
    # normative + speculative is a category error (caught by Pydantic)
    # but re-checked here for audit trail completeness
    (ClaimType.normative, EpistemicModality.asserted):      True,
    (ClaimType.normative, EpistemicModality.hedged):        True,
    (ClaimType.normative, EpistemicModality.speculative):   False,  # category error
    (ClaimType.normative, EpistemicModality.hypothetical):  True,
}

# Confidence language patterns that are inconsistent with asserted modality
# An asserted claim should have no hedging language
_HEDGING_MARKERS: frozenset[str] = frozenset({
    "likely", "probably", "possibly", "perhaps", "maybe",
    "appears", "seems", "suggests", "indicates", "may", "might",
    "could", "would", "approximately", "around", "roughly",
    "it is possible", "it seems", "it appears",
})

# Confidence language patterns inconsistent with speculative modality
# A speculative claim should not use certainty language
_CERTAINTY_MARKERS: frozenset[str] = frozenset({
    "certainly", "definitely", "clearly", "obviously", "undoubtedly",
    "it is clear", "it is certain", "without doubt", "absolutely",
    "always", "never", "all", "none",
})


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """
    Annotation produced by the classifier for a single Claim.

    is_valid:
        False only for true category errors (normative + speculative).
        All other combinations produce is_valid=True even with warnings.

    warnings:
        Consistency issues that do not make the claim invalid but
        indicate possible misclassification or internal contradiction.
        e.g. asserted modality with hedging confidence_language.

    suggestions:
        Optional reclassification suggestions from the classifier.
        The pipeline does not act on these automatically — they surface
        in the audit trail for human review.
    """

    claim_id: str
    is_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return not self.is_valid or bool(self.warnings)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ClaimClassifier:
    """
    Deterministic classifier for Step 2.

    Stateless — no constructor arguments.
    All classification logic is rule-based and defined in this module.
    """

    def classify(self, claim: Claim) -> ClassificationResult:
        """
        Classify a single claim for internal consistency.

        Parameters
        ----------
        claim:
            The extracted claim to classify.

        Returns
        -------
        ClassificationResult
            Always returns a result — never raises.
        """
        result = ClassificationResult(claim_id=claim.claim_id)

        self._check_type_modality_matrix(claim, result)
        self._check_confidence_language_consistency(claim, result)
        self._check_structural_preconditions(claim, result)
        self._check_scope_specificity(claim, result)

        return result

    def classify_all(self, claims: list[Claim]) -> list[ClassificationResult]:
        """
        Classify all claims. Returns results in the same order as input.
        """
        return [self.classify(claim) for claim in claims]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_type_modality_matrix(
        self,
        claim: Claim,
        result: ClassificationResult,
    ) -> None:
        """
        Check claim_type × epistemic_modality combination validity.
        """
        key = (claim.claim_type, claim.epistemic_modality)
        is_valid = _VALIDITY_MATRIX.get(key, True)

        if not is_valid:
            result.is_valid = False
            result.warnings.append(
                f"Category error: {claim.claim_type} + {claim.epistemic_modality} "
                f"is an invalid combination. Normative claims are prescriptions "
                f"and cannot be speculative. "
                f"Consider reclassifying as hypothetical if conditional."
            )

    def _check_confidence_language_consistency(
        self,
        claim: Claim,
        result: ClassificationResult,
    ) -> None:
        """
        Check confidence_language is consistent with epistemic_modality.

        An asserted claim with hedging language, or a speculative claim
        with certainty language, indicates the extraction agent assigned
        the wrong modality.
        """
        if not claim.confidence_language:
            return

        lang_lower = claim.confidence_language.lower()

        if claim.epistemic_modality == EpistemicModality.asserted:
            for marker in _HEDGING_MARKERS:
                if marker in lang_lower:
                    result.warnings.append(
                        f"Modality mismatch: claim is 'asserted' but "
                        f"confidence_language '{claim.confidence_language}' "
                        f"contains hedging marker '{marker}'. "
                        f"Consider reclassifying as 'hedged'."
                    )
                    result.suggestions.append(
                        f"Reclassify epistemic_modality as 'hedged'."
                    )
                    break  # one warning per claim is sufficient

        elif claim.epistemic_modality == EpistemicModality.speculative:
            for marker in _CERTAINTY_MARKERS:
                if marker in lang_lower:
                    result.warnings.append(
                        f"Modality mismatch: claim is 'speculative' but "
                        f"confidence_language '{claim.confidence_language}' "
                        f"contains certainty marker '{marker}'. "
                        f"Consider reclassifying as 'asserted' or 'hedged'."
                    )
                    result.suggestions.append(
                        f"Reclassify epistemic_modality as 'asserted' or 'hedged'."
                    )
                    break

    def _check_structural_preconditions(
        self,
        claim: Claim,
        result: ClassificationResult,
    ) -> None:
        """
        Re-validate structural preconditions with pipeline context.

        These are already enforced by Pydantic model validators, but
        re-checking here ensures they appear in the classification audit
        trail even if the Claim was constructed by bypassing validation
        (e.g. in tests using model_construct()).
        """
        if claim.claim_type == ClaimType.causal and not claim.assumptions:
            result.warnings.append(
                "Structural violation: causal claim has no assumptions. "
                "A causal mechanism must be declared. "
                "This claim will be assessed as inadmissible."
            )

        if claim.claim_type == ClaimType.predictive and claim.time_horizon is None:
            result.warnings.append(
                "Structural violation: predictive claim has no time_horizon. "
                "A prediction without a time horizon is unfalsifiable. "
                "This claim will be assessed as inadmissible."
            )

        if (
            claim.claim_type == ClaimType.predictive
            and claim.time_horizon is not None
            and not claim.time_horizon.is_quantified
        ):
            result.warnings.append(
                f"Soft warning: predictive claim has a vague time_horizon "
                f"('{claim.time_horizon.raw}' could not be normalised to ISO 8601). "
                f"The claim is admissible but epistemically weak."
            )

    def _check_scope_specificity(
        self,
        claim: Claim,
        result: ClassificationResult,
    ) -> None:
        """
        Warn when scope is suspiciously broad or explicitly unspecified.

        A claim with 'unspecified scope' was flagged by the extraction agent
        as having no determinable scope — this is a significant epistemic
        weakness for factual and causal claims.
        """
        if "unspecified scope" in claim.scope.lower():
            if claim.claim_type in (ClaimType.factual, ClaimType.causal):
                result.warnings.append(
                    f"Scope warning: {claim.claim_type} claim has unspecified scope. "
                    f"Factual and causal claims without scope cannot be "
                    f"verified against domain knowledge. "
                    f"The claim will be assessed as unsubstantiated."
                )