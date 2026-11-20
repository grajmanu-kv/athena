"""
core.protocols.admissibility
=============================
Protocol definition for domain admissibility checkers.

Admissibility checkers operate after guards (Step 3) and perform
the epistemic admissibility assessment (Step 4). Unlike guards — which
are deterministic and check structural/contract compliance — admissibility
checkers reason about whether a claim is epistemically credible given
available domain knowledge.

The SDK ships two built-in admissibility checkers:
    StructuralAdmissibilityChecker  — type × modality matrix rules
    ModalityAdmissibilityChecker    — confidence language consistency

Consuming systems extend this via their own domain checkers:
    e.g. a checker that queries a FIWARE NGSI-LD context broker
         to verify factual claims against live entity states.
    e.g. a checker that queries a CrateDB time-series store
         to verify whether a predictive claim is consistent
         with historical base rates.

Design contract
---------------
* A checker receives the full list of ClaimResults produced so far
  (after guards have run) plus the DecomposedPrompt and IntentContext.
  It may use other claims as evidence when assessing a specific claim
  (e.g. detecting contradiction between two claims in the same response).

* A checker MUST return an AdmissibilityAssessment for every claim.
  It must never raise.

* A checker MUST NOT assign AdmissibilityState.inadmissible on its own.
  inadmissible is reserved for structural guard failures. Checkers assign:
      verified | plausible | unsubstantiated | contradicted
  The assessor (stages/assess/assessor.py) synthesises all checker
  results and may escalate to inadmissible based on guard failures.

* Checkers are stateless between calls. Constructor injection for deps.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import AdmissibilityState, ClaimResult


# Private capability token. Only built-in structural checkers import it; a
# consumer checker cannot assign `inadmissible` by passing a public flag.
_BUILTIN = object()


class AdmissibilityAssessment(object):
    """
    The result of one admissibility checker's assessment of one claim.

    Not a Pydantic model — constructed and consumed within the pipeline
    only, never serialised directly. The final AdmissibilityState written
    to ClaimResult is synthesised from all checker assessments by the
    AdmissibilityAssessor.

    Parameters
    ----------
    checker_id:
        Stable identifier for the checker. Convention: class name.
    state:
        The admissibility state this checker assigns.
        Must not be AdmissibilityState.inadmissible (reserved for guards).
    rationale:
        Human-readable explanation. Required regardless of state.
    warnings:
        Optional soft warnings. They do not change `state`, but they make the
        claim count as 'with warnings' at the decision gate.
    evidence:
        Optional structured evidence supporting the rationale.
        e.g. {'matched_fact': 'Corridor A speed limit is 40 km/h',
               'claim_value': '40 km/h'}
    """

    __slots__ = ("checker_id", "state", "rationale", "evidence", "warnings", "_internal")

    def __init__(
        self,
        checker_id: str,
        state: AdmissibilityState,
        rationale: str,
        evidence: dict[str, str] | None = None,
        warnings: list[str] | None = None,
        _internal: object = None,
    ) -> None:
        # inadmissible is reserved for built-in structural checkers only.
        # Consumer-registered checkers must use contradicted instead.
        # Built-ins present the private _BUILTIN token to bypass this guard.
        if state == AdmissibilityState.inadmissible and _internal is not _BUILTIN:
            raise ValueError(
                f"Checker '{checker_id}' attempted to assign AdmissibilityState."
                "inadmissible. This state is reserved for built-in structural "
                "checkers. Use AdmissibilityState.contradicted for claims "
                "that conflict with known evidence."
            )
        self.checker_id = checker_id
        self.state = state
        self.rationale = rationale
        self.evidence: dict[str, str] = evidence or {}
        self.warnings: list[str] = list(warnings or [])
        self._internal = _internal is _BUILTIN

    def __repr__(self) -> str:
        return (
            f"AdmissibilityAssessment("
            f"checker_id={self.checker_id!r}, "
            f"state={self.state!r}, "
            f"rationale={self.rationale!r})"
        )


@runtime_checkable
class AdmissibilityCheckerProtocol(Protocol):
    """
    Protocol for domain admissibility checkers.

    Implementing this protocol is the only requirement for a checker to be
    registered with EpistemicValidator. No base class inheritance required.

    Example implementation:

        class NGSILDAdmissibilityChecker:
            '''Checks factual claims against a FIWARE NGSI-LD context broker.'''

            def __init__(self, broker_url: str) -> None:
                self._broker_url = broker_url

            @property
            def checker_id(self) -> str:
                return "NGSILDAdmissibilityChecker"

            async def assess(
                self,
                claim: Claim,
                all_claim_results: list[ClaimResult],
                decomposed_prompt: DecomposedPrompt,
                intent_context: IntentContext,
            ) -> AdmissibilityAssessment:
                if claim.claim_type != ClaimType.factual:
                    return AdmissibilityAssessment(
                        checker_id=self.checker_id,
                        state=AdmissibilityState.plausible,
                        rationale="Checker only evaluates factual claims. Skipped.",
                    )
                # ... query broker, compare entity attribute ...
                return AdmissibilityAssessment(
                    checker_id=self.checker_id,
                    state=AdmissibilityState.verified,
                    rationale="Entity attribute matches claim value.",
                    evidence={"entity_id": "urn:...", "attribute": "density"},
                )
    """

    @property
    def checker_id(self) -> str:
        """
        Stable string identifier for this checker.
        Used in audit trail entries.
        Convention: use the class name.
        """
        ...

    async def assess(
        self,
        claim: Claim,
        all_claim_results: list[ClaimResult],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> AdmissibilityAssessment:
        """
        Assess the epistemic admissibility of a single claim.

        Parameters
        ----------
        claim:
            The claim to assess. Do not mutate.
        all_claim_results:
            All ClaimResults produced so far in this validation run,
            including guard results for each. Available for cross-claim
            contradiction detection.
        decomposed_prompt:
            Structured prompt decomposition. Contains injected_facts,
            term_definitions, and constraint/boundary context.
        intent_context:
            The declared intent. Contains domain, action_class,
            consequence_class for risk-calibrated assessment.

        Returns
        -------
        AdmissibilityAssessment
            Always. Never raises.
            State must not be AdmissibilityState.inadmissible.
        """
        ...