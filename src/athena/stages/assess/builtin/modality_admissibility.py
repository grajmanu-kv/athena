"""
stages.assess.builtin.modality_admissibility
============================================
ModalityAdmissibilityChecker — adjusts AdmissibilityState based on
the epistemic modality of a claim relative to the declared risk level.

A claim's epistemic modality is not just a classification — it has
operational consequences. A `speculative` claim that is admissible in
a low-risk inform context may be inadmissible in a high-risk decide
context. This checker enforces that relationship.

Modality × RiskLevel admissibility matrix
------------------------------------------

                    low     medium      high    critical
asserted            P       P           P       P
hedged              P       P           P       P
speculative         P       W           F       F
hypothetical        P       W           W       F

P = plausible (no change to structural baseline)
W = warning  (plausible with explicit risk warning)
F = contributes to unsubstantiated (speculative/hypothetical claims
    in high-stakes contexts are epistemically insufficient)

Rationale
---------
A speculative claim in a critical/automate context — where the output
triggers automated system action with irreversible consequences —
is structurally inadmissible regardless of its content. The epistemic
modality itself is the problem: speculation cannot ground a critical
automated decision.

This is the decision-theoretic risk control principle from the spec:
the admissibility of information depends on the cost of error,
not only on likelihood.

Note: this checker does not assign inadmissible directly — that is
reserved for structural guard failures. It assigns unsubstantiated
for high-stakes speculative/hypothetical claims, which the assessor
may escalate to inadmissible based on aggregate results.
"""

from __future__ import annotations

from athena.core.risk.risk_level import derive_risk_level

from athena.core.models.claim import Claim, EpistemicModality
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import AdmissibilityState, ClaimResult, RiskLevel
from athena.core.protocols.admissibility import AdmissibilityAssessment


class ModalityAdmissibilityChecker:
    """
    Adjusts admissibility based on epistemic modality × risk level.

    Stateless — no constructor arguments.
    Depends on RiskLevel derived from IntentContext.
    """

    @property
    def checker_id(self) -> str:
        return "ModalityAdmissibilityChecker"

    async def assess(
        self,
        claim: Claim,
        all_claim_results: list[ClaimResult],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> AdmissibilityAssessment:

        risk_level = derive_risk_level(
            intent_context.action_class,
            intent_context.consequence_class,
        )

        modality = claim.epistemic_modality

        # asserted and hedged are admissible at all risk levels
        if modality in (EpistemicModality.asserted, EpistemicModality.hedged):
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=AdmissibilityState.plausible,
                rationale=(
                    f"Epistemic modality '{modality}' is admissible "
                    f"at risk level '{risk_level}'."
                ),
            )

        # speculative claims
        if modality == EpistemicModality.speculative:
            if risk_level == RiskLevel.low:
                return AdmissibilityAssessment(
                    checker_id=self.checker_id,
                    state=AdmissibilityState.plausible,
                    rationale=(
                        "Speculative claim at low risk level — admissible "
                        "as background exploration."
                    ),
                )
            if risk_level == RiskLevel.medium:
                return AdmissibilityAssessment(
                    checker_id=self.checker_id,
                    state=AdmissibilityState.plausible,
                    rationale=(
                        "Speculative claim at medium risk level — plausible "
                        "with warning. Speculation should not be the primary "
                        "basis for decisions."
                    ),
                    warnings=[
                        "Speculative claim at medium risk: flag explicitly to "
                        "the decision maker; not a basis for the decision."
                    ],
                    evidence={
                        "risk_level": risk_level,
                        "modality": modality,
                    },
                )
            # high or critical
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=AdmissibilityState.unsubstantiated,
                rationale=(
                    f"Speculative claim at {risk_level} risk level is "
                    f"epistemically insufficient. Speculation cannot ground "
                    f"a {risk_level}-stakes decision. "
                    f"The claim lacks the evidential basis required "
                    f"at this consequence class."
                ),
                evidence={
                    "risk_level": risk_level,
                    "modality": modality,
                    "action_class": intent_context.action_class,
                    "consequence_class": intent_context.consequence_class,
                },
            )

        # hypothetical claims
        if modality == EpistemicModality.hypothetical:
            if risk_level == RiskLevel.low:
                return AdmissibilityAssessment(
                    checker_id=self.checker_id,
                    state=AdmissibilityState.plausible,
                    rationale=(
                        "Hypothetical claim at low risk level — admissible "
                        "as conditional reasoning."
                    ),
                )
            if risk_level in (RiskLevel.medium, RiskLevel.high):
                return AdmissibilityAssessment(
                    checker_id=self.checker_id,
                    state=AdmissibilityState.plausible,
                    rationale=(
                        f"Hypothetical claim at {risk_level} risk level — "
                        f"plausible with warning. The unverified premise "
                        f"must be explicitly acknowledged by the decision maker."
                    ),
                    warnings=[
                        f"Hypothetical claim at {risk_level} risk: the "
                        f"unverified premise must be acknowledged."
                    ],
                    evidence={
                        "risk_level": risk_level,
                        "modality": modality,
                        "conditions": "; ".join(claim.conditions[:3]),
                    },
                )
            # critical
            return AdmissibilityAssessment(
                checker_id=self.checker_id,
                state=AdmissibilityState.unsubstantiated,
                rationale=(
                    f"Hypothetical claim at critical risk level is "
                    f"epistemically inadmissible. Conditional reasoning "
                    f"on unverified premises cannot ground critical decisions. "
                    f"The premise must be independently verified first."
                ),
                evidence={
                    "risk_level": risk_level,
                    "modality": modality,
                    "conditions": "; ".join(claim.conditions[:3]),
                },
            )

        # Unreachable — all EpistemicModality values handled
        return AdmissibilityAssessment(
            checker_id=self.checker_id,
            state=AdmissibilityState.plausible,
            rationale=f"Unhandled modality '{modality}' — defaulting to plausible.",
        )