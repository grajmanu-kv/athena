"""
core.models.decomposed_prompt
==============================
The structured artifact produced by Step 0 (prompt decomposition).

After Step 0, the raw prompt string is never accessed again by the pipeline.
All subsequent stages operate on the DecomposedPrompt produced here.

Design notes
------------
* explicit_* fields are populated directly from DomainInfo (declared by
  the consuming system). implicit_* fields are extracted from the raw
  prompt by the decomposition agent.

* The decomposition agent does not replace declared constraints/boundaries
  — it augments them. explicit_ always takes precedence over implicit_ in
  guard evaluation.

* injected_facts are data embedded in the prompt itself (sensor readings,
  retrieved documents, prior context). They are distinct from
  DomainInfo.known_facts (standing domain knowledge). Both feed Step 4
  but with different trust levels: known_facts are ground truth declared
  by the consuming system; injected_facts are whatever the prompt happened
  to include and should be treated as contextual, not authoritative.

* undeclared_terms are technical terms found in the prompt or response
  that have no definition in DomainInfo.term_definitions and could not be
  inferred from the prompt. These are flagged by the terminology guard as
  potential sources of semantic drift.

* extraction_notes is an optional free-text field the decomposition agent
  may populate to explain ambiguous decomposition decisions. It is
  included verbatim in the audit trail.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class DecomposedPrompt(BaseModel):
    """
    Structured decomposition of the raw prompt produced by Step 0.

    This object is the epistemic contract against which all claims
    extracted in Step 1 are subsequently validated.

    Field routing through pipeline stages:
        explicit_constraints + implicit_constraints  → Step 3 ConstraintGuard
        explicit_boundaries  + implicit_boundaries   → Step 3 BoundaryGuard
        term_definitions                             → Step 3 TerminologyGuard
        undeclared_terms                             → Step 3 TerminologyGuard (flag)
        injected_facts                               → Step 4 AdmissibilityAssessor
        extraction_notes                             → AuditReport
    """

    model_config = {"frozen": True}

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    explicit_constraints: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Constraints passed directly from DomainInfo.constraints. "
                "These are declared by the consuming system and are "
                "authoritative — the LLM was explicitly required to satisfy them."
            ),
        ),
    ]

    implicit_constraints: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Constraints extracted from the raw prompt by the decomposition "
                "agent that were not declared in DomainInfo.constraints. "
                "e.g. a prompt beginning 'As a cautious advisor, always...' "
                "implies a constraint on epistemic hedging. "
                "Lower trust than explicit_constraints — flagged distinctly "
                "in the audit trail."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Boundaries
    # ------------------------------------------------------------------

    explicit_boundaries: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Boundaries passed directly from DomainInfo.boundaries. "
                "Authoritative — the LLM was explicitly forbidden these."
            ),
        ),
    ]

    implicit_boundaries: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Boundaries extracted from the raw prompt by the decomposition "
                "agent that were not declared in DomainInfo.boundaries. "
                "e.g. a role framing like 'You must not speculate' embeds "
                "a boundary the consuming system did not declare explicitly."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Term definitions
    # ------------------------------------------------------------------

    term_definitions: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description=(
                "Merged term definitions: DomainInfo.term_definitions (authoritative) "
                "augmented by any additional definitions extracted from the prompt. "
                "DomainInfo definitions always win on collision. "
                "All claims are validated against these definitions — not against "
                "everyday usage or LLM training distribution semantics."
            ),
        ),
    ]

    undeclared_terms: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Technical terms used in the prompt or response that have no "
                "definition in term_definitions and whose meaning could not be "
                "inferred from context. "
                "Flagged by the terminology guard as potential semantic drift "
                "vectors. Not a hard failure — a soft warning in the audit trail."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Injected facts
    # ------------------------------------------------------------------

    injected_facts: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Data embedded in the prompt itself: sensor readings, "
                "retrieved documents, prior conversation turns, entity states. "
                "Distinct from DomainInfo.known_facts (standing domain knowledge). "
                "Trust level: contextual — provided for this invocation, "
                "not standing ground truth declared by the consuming system."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Decomposition metadata
    # ------------------------------------------------------------------

    extraction_notes: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional free-text notes from the decomposition agent "
                "explaining ambiguous or uncertain decomposition decisions. "
                "Included verbatim in the audit trail. "
                "e.g. 'Could not determine whether the injected sensor data "
                "was current or historical — treated as contextual.'"
            ),
        ),
    ] = None

    # ------------------------------------------------------------------
    # LLM response — stored for response-level guard evaluation
    # ------------------------------------------------------------------

    llm_response: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "The raw LLM response text. Stored here so guards that need "
                "response-level context (e.g. ConstraintGuard evaluating citation "
                "constraints) can access the full text without changing the "
                "GuardProtocol signature. Populated by EpistemicValidator "
                "after decomposition."
            ),
        ),
    ] = None

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def all_constraints(self) -> list[str]:
        """Union of explicit and implicit constraints for guard evaluation."""
        return self.explicit_constraints + self.implicit_constraints

    @property
    def all_boundaries(self) -> list[str]:
        """Union of explicit and implicit boundaries for guard evaluation."""
        return self.explicit_boundaries + self.implicit_boundaries

    @property
    def has_any_contract(self) -> bool:
        """
        True when at least one constraint or boundary is available
        for guard evaluation (explicit or implicit).
        """
        return bool(self.all_constraints or self.all_boundaries)

    @property
    def has_undeclared_terms(self) -> bool:
        """True when the terminology guard has terms to flag."""
        return bool(self.undeclared_terms)