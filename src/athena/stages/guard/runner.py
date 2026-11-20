"""
stages.guard.runner
===================
Step 3 — Guard Execution.

Runs all registered guards against each claim concurrently.
Guards within a single claim run in parallel via asyncio.gather.
Claims themselves are processed sequentially to preserve order
and avoid race conditions in consumer-provided guards that may
share state.

Execution model
---------------
For N claims and M guards:
    - M guards run concurrently per claim  (asyncio.gather)
    - N claims run sequentially            (for loop)

This gives O(N × max_guard_latency) rather than O(N × M × guard_latency).
For purely deterministic guards (all built-ins), concurrency is irrelevant
but the model is correct for consumer guards that may make async I/O calls.

Exception handling
------------------
Guards must never raise — the protocol contract requires them to always
return a GuardResult. However, a buggy guard implementation may raise
despite this contract. GuardRunner catches all exceptions from guard.evaluate()
calls and converts them to a GuardResult with outcome=failed and the
exception message in the detail field. This prevents a single bad guard
from aborting the entire pipeline.

The guard that raised is logged via loguru. The consuming system can
inspect GuardResult.detail['exception'] to identify the failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import GuardOutcome, GuardResult
from athena.core.protocols.guard import GuardProtocol


# ---------------------------------------------------------------------------
# Guard run record — internal to runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardRunRecord:
    """
    Internal record of a guard's execution against a single claim.
    Collected by the runner and attached to ClaimResult in Step 4.
    """

    claim_id: str
    guard_results: list[GuardResult]

    @property
    def has_failures(self) -> bool:
        return any(r.outcome == GuardOutcome.failed for r in self.guard_results)

    @property
    def has_warnings(self) -> bool:
        return any(
            r.outcome in (GuardOutcome.warning, GuardOutcome.errored)
            for r in self.guard_results
        )

    @property
    def has_errors(self) -> bool:
        return any(r.outcome == GuardOutcome.errored for r in self.guard_results)


# ---------------------------------------------------------------------------
# GuardRunner
# ---------------------------------------------------------------------------


_MAX_CONCURRENT_CLAIMS = 8


class GuardRunner:
    """
    Async parallel executor for all registered guards.

    Parameters
    ----------
    guards:
        Ordered list of GuardProtocol implementations.
        Built-in guards should be registered before consumer guards.
    """

    def __init__(self, guards: list[GuardProtocol]) -> None:
        if not guards:
            logger.warning(
                "GuardRunner initialised with no guards. "
                "No constraint or boundary checking will occur."
            )
        self._guards = guards

    async def run_all(
        self,
        claims: list[Claim],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> list[GuardRunRecord]:
        """
        Run all guards against all claims.

        Parameters
        ----------
        claims:
            All extracted claims.
        decomposed_prompt:
            Structured prompt decomposition. Passed to each guard.
        intent_context:
            Declared intent. Passed to each guard.

        Returns
        -------
        list[GuardRunRecord]
            One record per claim, in the same order as input claims.
            Each record contains GuardResults from all guards.
        """
        # Claims are independent: evaluate them concurrently (bounded, so a
        # large response cannot open unbounded parallel LLM calls). gather()
        # preserves input order.
        sem = asyncio.Semaphore(_MAX_CONCURRENT_CLAIMS)

        async def _one(claim: Claim) -> GuardRunRecord:
            async with sem:
                guard_results = await self._run_guards_for_claim(
                    claim, decomposed_prompt, intent_context
                )
            return GuardRunRecord(claim_id=claim.claim_id, guard_results=guard_results)

        return list(await asyncio.gather(*(_one(c) for c in claims)))

    async def _run_guards_for_claim(
        self,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> list[GuardResult]:
        """
        Run all guards concurrently for a single claim.

        All exceptions from guard.evaluate() are caught and converted
        to `errored` GuardResults — a buggy guard cannot abort the pipeline,
        and a crash is not misreported as a failed check on the claim.
        """
        tasks = [
            self._safe_evaluate(guard, claim, decomposed_prompt, intent_context)
            for guard in self._guards
        ]
        results: list[GuardResult] = await asyncio.gather(*tasks)
        return list(results)

    async def _safe_evaluate(
        self,
        guard: GuardProtocol,
        claim: Claim,
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> GuardResult:
        """
        Wrap guard.evaluate() in exception handling.

        Returns an errored GuardResult if the guard raises unexpectedly.
        """
        try:
            return await guard.evaluate(claim, decomposed_prompt, intent_context)
        except Exception as exc:
            logger.error(
                "Guard '{}' raised an unexpected exception for claim '{}': {}. "
                "This indicates a bug in the guard implementation. "
                "The guard contract requires evaluate() to never raise.",
                guard.guard_id,
                claim.claim_id,
                exc,
            )
            return GuardResult(
                guard_id=guard.guard_id,
                outcome=GuardOutcome.errored,
                message=(
                    f"Guard raised an unexpected exception: {type(exc).__name__}. "
                    f"This is a guard implementation bug, not a claim failure. "
                    f"See pipeline logs for the full traceback."
                ),
                detail={
                    "exception": type(exc).__name__,
                    "exception_message": str(exc),
                    "claim_id": claim.claim_id,
                },
            )