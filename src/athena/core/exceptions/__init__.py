"""
core.exceptions
===============
All exceptions raised by the Athena epistemic validation library.

Exception hierarchy:

    AthenaError                         — base for all library exceptions
    ├── EpistemicValidationError        — validation pipeline failures
    │   ├── ExtractionFailure           — Step 0 or Step 1 LLM call failed
    │   │   ├── DecompositionFailure    — Step 0 prompt decomposition failed
    │   │   └── ClaimExtractionFailure  — Step 1 claim extraction failed
    │   ├── GuardConfigurationError     — guard registered incorrectly
    │   └── DependencyGraphCycleError   — cycle detected in claim graph
    └── AthenaConfigurationError        — validator constructed incorrectly

Design notes
------------
* All exceptions carry a human-readable message as the first argument.
* Exceptions that wrap an underlying cause always chain via `raise X from cause`.
* Pipeline stages catch exceptions from LLM calls and re-raise as the
  appropriate subclass — consuming systems never see raw httpx or
  pydantic_ai exceptions from within the pipeline.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class AthenaError(Exception):
    """
    Base exception for all errors raised by the Athena library.
    Consuming systems may catch this to handle any Athena error uniformly.
    """


# ---------------------------------------------------------------------------
# Validation errors — raised during pipeline execution
# ---------------------------------------------------------------------------


class EpistemicValidationError(AthenaError):
    """
    Raised when the validation pipeline encounters an unrecoverable error.

    Distinct from a validation *result* that contains inadmissible claims —
    that is a normal outcome represented by ValidationResult.verdict.
    This exception signals that the pipeline itself could not complete.
    """


class ExtractionFailure(EpistemicValidationError):
    """
    Raised when an LLM-based extraction stage fails unrecoverably.

    This covers both retry exhaustion (the LLM repeatedly returned
    schema-invalid output) and LLM provider errors (network failure,
    rate limit, context length exceeded).

    Attributes
    ----------
    stage:
        Which pipeline stage failed. 'decompose' or 'extract'.
    attempts:
        Number of attempts made before giving up.
    last_error:
        The underlying exception from the last attempt, if available.
    """

    def __init__(
        self,
        message: str,
        stage: str,
        attempts: int = 1,
        last_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts
        self.last_error = last_error

    def __str__(self) -> str:
        base = super().__str__()
        return (
            f"{base} "
            f"[stage={self.stage}, attempts={self.attempts}"
            f"{', last_error=' + repr(self.last_error) if self.last_error else ''}]"
        )


class DecompositionFailure(ExtractionFailure):
    """
    Raised when Step 0 (prompt decomposition) fails unrecoverably.
    The raw prompt could not be structured into a DecomposedPrompt.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 1,
        last_error: BaseException | None = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="decompose",
            attempts=attempts,
            last_error=last_error,
        )


class ClaimExtractionFailure(ExtractionFailure):
    """
    Raised when Step 1 (claim extraction) fails unrecoverably.
    The LLM response could not be decomposed into a list of Claims.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 1,
        last_error: BaseException | None = None,
    ) -> None:
        super().__init__(
            message=message,
            stage="extract",
            attempts=attempts,
            last_error=last_error,
        )


class GuardConfigurationError(EpistemicValidationError):
    """
    Raised when a guard is registered incorrectly with EpistemicValidator.

    Causes:
    - Guard does not implement GuardProtocol
    - Two guards registered with the same guard_id
    - Guard raises during evaluate() in a way that is not caught
      (indicates a bug in the guard implementation)
    """


class DependencyGraphCycleError(EpistemicValidationError):
    """
    Raised when the dependency graph builder detects a cycle.

    A cycle in the depends_on graph indicates that compound claims
    were not properly decomposed at extraction time. The cycle
    makes topological ordering — and therefore failure propagation —
    impossible.

    The error message identifies the cycle where possible.
    """


# ---------------------------------------------------------------------------
# Configuration errors — raised at construction time
# ---------------------------------------------------------------------------


class AthenaConfigurationError(AthenaError):
    """
    Raised when EpistemicValidator is constructed with invalid configuration.

    Causes:
    - No model provided
    - Incompatible admissibility_checker registered
    - Invalid dependency_builder provided
    - Conflicting guard_ids in the guard list
    """