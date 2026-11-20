"""
core.protocols.guard
====================
Protocol definition for epistemic guards.

Guards are deterministic, synchronous or asynchronous validators that
check a single Claim against the DecomposedPrompt and IntentContext.
They never call an LLM. They never mutate their inputs.

Three categories of guards exist in the pipeline:

    Built-in fixed guards (always run, non-overridable):
        StructuralGuard     — schema completeness per ClaimType
        ConstraintGuard     — checks against DecomposedPrompt.all_constraints
        BoundaryGuard       — checks against DecomposedPrompt.all_boundaries
        TerminologyGuard    — checks term usage against term_definitions

    Consumer-provided semantic guards (optional, registered at construction):
        Any class implementing GuardProtocol.
        e.g. a domain guard that checks claims against a live database.

All guards — built-in and consumer — implement the same protocol.
The GuardRunner (stages/guard/runner.py) executes them uniformly.

Design contract
---------------
* A guard MUST return a GuardResult for every claim it receives.
  It must never raise — exceptions are caught by GuardRunner and
  converted to a failed GuardResult with the exception message.

* A guard MUST set outcome=GuardOutcome.skipped when the claim type
  or modality makes the guard inapplicable, rather than returning
  a spurious pass.

* A guard MUST populate GuardResult.message regardless of outcome.
  The audit trail requires an explanation for every decision including passes.

* Guards are stateless between calls. Any state required for evaluation
  must be provided via constructor injection.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import GuardResult


@runtime_checkable
class GuardProtocol(Protocol):
    """
    Protocol for all epistemic guards in the Athena pipeline.

    Implementing this protocol is the only requirement for a guard to be
    registered with EpistemicValidator. No base class inheritance required.

    Example implementation:

        class MyDomainGuard:
            '''Checks claims against a live entity database.'''

            def __init__(self, entity_store: EntityStore) -> None:
                self._store = entity_store

            @property
            def guard_id(self) -> str:
                return "MyDomainGuard"

            async def evaluate(
                self,
                claim: Claim,
                decomposed_prompt: DecomposedPrompt,
                intent_context: IntentContext,
            ) -> GuardResult:
                entity = await self._store.get(claim.scope)
                if entity is None:
                    return GuardResult(
                        guard_id=self.guard_id,
                        outcome=GuardOutcome.warning,
                        message=f"Scope '{claim.scope}' not found in entity store.",
                        detail={"scope": claim.scope},
                    )
                return GuardResult(
                    guard_id=self.guard_id,
                    outcome=GuardOutcome.passed,
                    message=f"Scope '{claim.scope}' verified in entity store.",
                    detail={},
                )
    """

    @property
    def guard_id(self) -> str:
        """
        Stable string identifier for this guard.
        Used in GuardResult.guard_id and audit trail entries.
        Must be unique within a validator instance.
        Convention: use the class name.
        """
        ...

    async def evaluate(
        self,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> GuardResult:
        """
        Evaluate a single claim and return a GuardResult.

        Parameters
        ----------
        claim:
            The extracted, classified claim to evaluate.
            Do not mutate.
        decomposed_prompt:
            The structured decomposition of the raw prompt.
            Contains all_constraints, all_boundaries, term_definitions,
            injected_facts, and undeclared_terms.
        intent_context:
            The declared intent of the calling system.
            Contains action_class, consequence_class, domain,
            and decision_window.

        Returns
        -------
        GuardResult
            Always. Never raises. Use outcome=GuardOutcome.skipped
            when the guard is not applicable to this claim.
        """
        ...