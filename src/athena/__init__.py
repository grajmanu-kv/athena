"""
athena
======
Epistemic Verification Framework for LLM Outputs.

Public API available via direct import:

    from athena import EpistemicValidator, ValidationInput, ...

Or via lazy access — pydantic_ai is only imported when EpistemicValidator
is first accessed, allowing core models and utilities to be imported
without requiring pydantic_ai to be installed.
"""

from __future__ import annotations

# Core models — no pydantic_ai dependency
from athena.core.models.input import (
    ActionClass,
    ConsequenceClass,
    DomainInfo,
    IntentContext,
    ValidationInput,
)
from athena.core.models.claim import (
    Claim,
    ClaimType,
    EpistemicModality,
    TimeHorizon,
)
from athena.core.models.result import (
    AdmissibilityState,
    AuditReport,
    ClaimResult,
    GuardOutcome,
    GuardResult,
    RiskLevel,
    ValidationResult,
    Verdict,
    VerdictCode,
)
from athena.core.protocols.admissibility import (
    AdmissibilityAssessment,
    AdmissibilityCheckerProtocol,
)
from athena.core.protocols.dependency import (
    DependencyGraph,
    DependencyGraphBuilderProtocol,
)
from athena.core.protocols.guard import GuardProtocol
from athena.core.exceptions import (
    AthenaConfigurationError,
    AthenaError,
    ClaimExtractionFailure,
    DecompositionFailure,
    DependencyGraphCycleError,
    EpistemicValidationError,
    ExtractionFailure,
    GuardConfigurationError,
)
from athena.core.risk.risk_level import derive_risk_level


def __getattr__(name: str) -> object:
    """
    Lazy import for EpistemicValidator.

    pydantic_ai is only imported when EpistemicValidator is first accessed.
    This allows core models, protocols, and utilities to be imported
    in test environments or contexts where pydantic_ai is not installed.
    """
    if name == "EpistemicValidator":
        from athena.pipeline.validator import EpistemicValidator
        return EpistemicValidator
    raise AttributeError(f"module 'athena' has no attribute {name!r}")


__all__ = [
    # Entry point (lazy)
    "EpistemicValidator",
    # Input
    "ValidationInput",
    "IntentContext",
    "DomainInfo",
    "ActionClass",
    "ConsequenceClass",
    # Claim
    "Claim",
    "ClaimType",
    "EpistemicModality",
    "TimeHorizon",
    # Results
    "ValidationResult",
    "ClaimResult",
    "GuardResult",
    "GuardOutcome",
    "Verdict",
    "VerdictCode",
    "AdmissibilityState",
    "RiskLevel",
    "AuditReport",
    # Protocols
    "GuardProtocol",
    "AdmissibilityCheckerProtocol",
    "AdmissibilityAssessment",
    "DependencyGraphBuilderProtocol",
    "DependencyGraph",
    # Utilities
    "derive_risk_level",
    # Exceptions
    "AthenaError",
    "EpistemicValidationError",
    "ExtractionFailure",
    "DecompositionFailure",
    "ClaimExtractionFailure",
    "GuardConfigurationError",
    "DependencyGraphCycleError",
    "AthenaConfigurationError",
]