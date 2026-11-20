"""
core.models.claim
=================
All types that constitute an epistemic claim extracted from an LLM response.

Design notes
------------
* ClaimType and EpistemicModality are orthogonal axes — every Claim carries both.
  ClaimType answers "what kind of proposition is this?"
  EpistemicModality answers "with what epistemic commitment is it stated?"

* TimeHorizon is a two-field structure: `raw` preserves the verbatim source text
  exactly as the LLM produced it; `normalized` holds an ISO 8601 duration string
  if the extractor could unambiguously parse one, otherwise None.
  Guards use the presence/absence of `normalized` to distinguish between
  a quantified horizon (full validity check) and a vague horizon (soft warning).

* source_span is mandatory and non-nullable. A claim without a traceable source
  fragment is epistemically unanchored — the same failure mode the library exists
  to prevent.

* depends_on and negates are empty at extraction time. They are populated
  exclusively by the dependency resolution stage (Step 5), which returns
  annotated copies of each Claim (Claims themselves are frozen). Downstream
  stages must not write to them.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ClaimType(StrEnum):
    """
    Epistemic type of a claim — what kind of proposition it makes.

    factual     : asserts a state of the world that is in principle verifiable
                  against evidence.  "Vehicle density on Corridor A is 87 veh/km."

    causal      : asserts a causal relationship between two states of the world.
                  Requires at least one assumption (the proposed mechanism).
                  "The congestion on Corridor A was caused by the signal fault."

    predictive  : asserts a future state of the world.
                  Requires a time_horizon (quantified or vague).
                  "Density will exceed 100 veh/km within the next 15 minutes."

    normative   : asserts what should be done or what is desirable.
                  Cannot be empirically scored — guards treat it differently.
                  "The operator should trigger a manual override immediately."
    """

    factual     = "factual"
    causal      = "causal"
    predictive  = "predictive"
    normative   = "normative"


class EpistemicModality(StrEnum):
    """
    Epistemic modality of a claim — with what commitment is it stated.

    asserted    : stated as established fact, no qualification.
                  Triggers the strictest guard checks.

    hedged      : stated with explicit uncertainty language
                  ("likely", "probably", "appears to").
                  Weakens guard thresholds but does not disable them.

    speculative : exploratory reasoning with no factual anchor.
                  Cannot support downstream asserted claims in the
                  dependency graph.

    hypothetical: conditional on an unstated or unverified premise
                  ("if X were true, then Y").
                  The premise itself must be extracted as a condition.
    """

    asserted     = "asserted"
    hedged       = "hedged"
    speculative  = "speculative"
    hypothetical = "hypothetical"


# ---------------------------------------------------------------------------
# TimeHorizon
# ---------------------------------------------------------------------------


# ISO 8601 duration with at least one component (see TimeHorizon.normalized).
_TIME = r"T(?:\d+H(?:\d+M)?(?:\d+S)?|\d+M(?:\d+S)?|\d+S)"
_DATE = r"(?:\d+Y(?:\d+M)?(?:\d+W)?(?:\d+D)?|\d+M(?:\d+W)?(?:\d+D)?|\d+W(?:\d+D)?|\d+D)"
_ISO_DURATION_PATTERN = rf"^P(?:{_DATE}(?:{_TIME})?|{_TIME})$|^None$"


class TimeHorizon(BaseModel):
    """
    Temporal scope of a predictive claim.

    raw         : verbatim fragment from the source response.
                  Always populated for predictive claims.
                  e.g. "in the next 15 minutes", "by end of peak hour",
                       "in the near future"

    normalized  : ISO 8601 duration string if the extractor could unambiguously
                  parse one from `raw`, otherwise None.
                  e.g. "PT15M", "PT1H", None

    Guard logic:
        normalized is not None  → full validity check
                                  (horizon within the declared decision window,
                                  when the window is itself a parseable ISO 8601
                                  duration; see StructuralGuard)
        normalized is None      → soft warning raised, presence check passes
        TimeHorizon absent      → hard guard failure for predictive claims
    """

    model_config = {"frozen": True}

    raw: Annotated[
        str,
        Field(
            min_length=1,
            description="Verbatim time reference from the LLM response.",
        ),
    ]

    normalized: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "ISO 8601 duration (e.g. 'PT15M') if unambiguously parseable "
                "from raw, otherwise None."
            ),
            # At least one component is required: bare "P", "PT" and a dangling
            # "T" are not durations. (No look-ahead: pydantic's regex engine
            # does not support it, so the alternatives are spelled out.)
            pattern=_ISO_DURATION_PATTERN,
        ),
    ] = None

    @field_validator("normalized", mode="before")
    @classmethod
    def coerce_none_string(cls, v: object) -> str | None:
        """Treat the string literal 'None' as Python None."""
        if v == "None":
            return None
        return v  # type: ignore[return-value]

    @property
    def is_quantified(self) -> bool:
        """True when the horizon is expressible as a concrete duration."""
        return self.normalized is not None


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """
    The minimal epistemic unit suitable for scientific analysis.

    A Claim is an atomic proposition extracted from an LLM response that
    commits the speaker to a state of the world being true, false, likely,
    or desirable under stated or implied conditions.

    Atomicity constraint: a Claim must not contain more than one independent
    proposition. Compound propositions connected by "and", "but", or "however"
    must be split at extraction time.

    Immutability: Claims are frozen after construction. Pipeline stages
    annotate them by producing ClaimResult wrappers (see result.py),
    never by mutating the Claim itself.
    """

    model_config = {"frozen": True}

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    claim_id: Annotated[
        str,
        Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="UUID4 identifier. Auto-generated at extraction time.",
        ),
    ]

    # ------------------------------------------------------------------
    # Epistemic classification
    # ------------------------------------------------------------------

    claim_type: Annotated[
        ClaimType,
        Field(description="What kind of proposition this claim makes."),
    ]

    epistemic_modality: Annotated[
        EpistemicModality,
        Field(description="With what epistemic commitment this claim is stated."),
    ]

    # ------------------------------------------------------------------
    # Propositional content
    # ------------------------------------------------------------------

    proposition: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The atomic claim restated in plain unambiguous language. "
                "Must not contain conjunctions linking independent propositions."
            ),
        ),
    ]

    source_span: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Verbatim fragment from the LLM response from which this "
                "claim was extracted. Mandatory for audit traceability. "
                "Must appear verbatim in the original response."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Epistemic conditions
    # ------------------------------------------------------------------

    conditions: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Conditions under which this claim holds. "
                "Empty list means the claim is asserted unconditionally. "
                "e.g. ['if sensor CAM-042 is functioning correctly']"
            ),
        ),
    ]

    assumptions: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Presuppositions the claim requires to be valid. "
                "Non-empty required for causal claims (the proposed mechanism). "
                "e.g. ['traffic pattern has not changed since last calibration']"
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    scope: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Domain, temporal, and spatial scope of this claim. "
                "e.g. 'Corridor A, 14:32-14:35 IST, 21 March 2026'"
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Confidence and time
    # ------------------------------------------------------------------

    confidence_language: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Verbatim confidence marker from the source text. "
                "e.g. 'likely', 'certainly', 'may', 'it is clear that'. "
                "None if no explicit confidence language is present. "
                "Distinct from epistemic_modality: this is raw evidence; "
                "modality is the library's derived classification."
            ),
        ),
    ] = None

    time_horizon: Annotated[
        TimeHorizon | None,
        Field(
            default=None,
            description=(
                "Temporal horizon for predictive claims. "
                "Required non-None for ClaimType.predictive. "
                "Hard guard failure if absent on a predictive claim."
            ),
        ),
    ] = None

    # ------------------------------------------------------------------
    # Dependency graph edges — populated by resolve stage only
    # ------------------------------------------------------------------

    depends_on: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "claim_ids this claim logically presupposes. "
                "Empty at extraction time. "
                "Populated exclusively by the dependency resolution stage."
            ),
        ),
    ]

    negates: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "claim_ids this claim logically contradicts. "
                "Empty at extraction time. "
                "Populated exclusively by the dependency resolution stage."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("assumptions", "conditions", mode="after")
    @classmethod
    def drop_blank_entries(cls, v: list[str]) -> list[str]:
        """
        Strip whitespace and discard blank entries.

        A blank string is not an assumption: without this, ``assumptions=[" "]``
        would satisfy the 'no causality without assumptions' invariant.
        """
        return [item.strip() for item in v if item and item.strip()]

    @model_validator(mode="after")
    def causal_requires_assumptions(self) -> "Claim":
        """
        A causal claim with no assumptions is epistemically incomplete.
        The assumptions field must document the proposed causal mechanism.
        """
        if self.claim_type == ClaimType.causal and not self.assumptions:
            raise ValueError(
                f"Causal claim '{self.claim_id}' must declare at least one "
                "assumption documenting the proposed causal mechanism. "
                "A causal claim without a mechanism is unfalsifiable."
            )
        return self

    @model_validator(mode="after")
    def predictive_requires_time_horizon(self) -> "Claim":
        """
        A predictive claim without a time horizon cannot be falsified.
        Popper: a statement that cannot be tested in principle is not
        scientifically evaluable.
        """
        if self.claim_type == ClaimType.predictive and self.time_horizon is None:
            raise ValueError(
                f"Predictive claim '{self.claim_id}' must declare a time_horizon. "
                "A prediction without a time horizon is unfalsifiable and "
                "therefore inadmissible."
            )
        return self

    @model_validator(mode="after")
    def normative_modality_consistency(self) -> "Claim":
        """
        A normative claim stated as speculative is a category confusion —
        normative claims are prescriptions, not speculations about the world.
        Hypothetical normative claims are valid: 'if X, one should do Y'.
        """
        if (
            self.claim_type == ClaimType.normative
            and self.epistemic_modality == EpistemicModality.speculative
        ):
            raise ValueError(
                f"Normative claim '{self.claim_id}' cannot have modality "
                "'speculative'. Normative claims are prescriptions. "
                "Use 'hypothetical' for conditional prescriptions."
            )
        return self

    @model_validator(mode="after")
    def dependency_ids_not_self_referential(self) -> "Claim":
        """A claim cannot depend on or negate itself."""
        if self.claim_id in self.depends_on:
            raise ValueError(
                f"Claim '{self.claim_id}' lists itself in depends_on."
            )
        if self.claim_id in self.negates:
            raise ValueError(
                f"Claim '{self.claim_id}' lists itself in negates."
            )
        return self