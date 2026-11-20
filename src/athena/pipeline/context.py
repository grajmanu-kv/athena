"""
pipeline.context
================
PipelineContext — internal state carrier threaded through all pipeline stages.

Never exposed to consuming systems. All public API inputs and outputs
use the types defined in core.models.

The context carries:
- The original ValidationInput (immutable reference)
- Intermediate artifacts produced by each stage
- The run_id for log correlation
- Pipeline-level warnings collected across all stages

Design: a plain dataclass with mutable fields. Pipeline stages
write to it sequentially — there is no concurrent write access
because the pipeline is orchestrated sequentially in validator.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from athena.core.models.claim import Claim
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import ValidationInput
from athena.core.models.result import ClaimResult
from athena.core.protocols.dependency import DependencyGraph
from athena.stages.classifier.agent import ClassificationResult
from athena.stages.extract.agent import ExtractionWarning


@dataclass
class PipelineContext:
    """
    Internal mutable state carrier for a single validation run.

    Stages write their outputs here. The validator reads the final
    state to assemble ValidationResult.
    """

    # Inputs — set at construction, never mutated
    validation_input: ValidationInput
    run_id: str

    # Stage 0 output
    decomposed_prompt: DecomposedPrompt | None = None

    # Stage 1 output
    extracted_claims: list[Claim] = field(default_factory=list)
    extraction_warnings: list[ExtractionWarning] = field(default_factory=list)

    # Stage 2 output
    classification_results: list[ClassificationResult] = field(
        default_factory=list
    )

    # Stage 3 output — claim_id → list[GuardResult]
    # (GuardResult imported lazily to avoid circular at module level)
    guard_results_map: dict = field(default_factory=dict)

    # Stage 4 output
    assessed_claim_results: list[ClaimResult] = field(default_factory=list)

    # Stage 5 output
    dependency_graph: DependencyGraph | None = None
    resolved_claim_results: list[ClaimResult] = field(default_factory=list)

    # Pipeline-level warnings accumulated across all stages
    pipeline_warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        """Add a pipeline-level warning to the audit trail."""
        self.pipeline_warnings.append(message)

    @property
    def final_claim_results(self) -> list[ClaimResult]:
        """
        The definitive ClaimResults after all stages.
        Prefers resolved (post-propagation) over assessed (pre-propagation).
        """
        return self.resolved_claim_results or self.assessed_claim_results