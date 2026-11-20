"""
stages.resolve.builder
=======================
HeuristicDependencyBuilder — default dependency graph builder.

Infers claim dependencies using two heuristic strategies without
an LLM call:

Strategy 1 — Claim type ordering rules
----------------------------------------
Some claim type combinations have a natural epistemic dependency order:

    causal → factual:   A causal claim "X caused Y" presupposes
                        the factual claim "Y occurred" or "X occurred".
                        If a factual claim in the same response asserts
                        one of the causal claim's referents, the causal
                        claim depends on it.

    predictive → factual: A prediction "X will increase" often builds
                          on a current factual observation "X is currently Y".
                          If a factual claim describes the current state
                          referenced by a prediction, the prediction depends
                          on it.

    normative → factual/causal: A recommendation "we should do X"
                                typically depends on a causal or factual
                                observation that grounds it.

Strategy 2 — Proposition text similarity (scope matching)
----------------------------------------------------------
Two claims that share the same scope string likely refer to the same
entity/event. If one is factual and the other is causal or predictive,
the causal/predictive claim depends on the factual one.

Strategy 3 — Negation detection
---------------------------------
Two claims that explicitly negate each other are detected by checking
for negation markers in opposing propositions:
- "X is high" vs "X is not high" or "X is low"
- Uses a simple keyword comparison — not semantic NLI

Limitations
-----------
Heuristic dependency inference is inherently imprecise. It will produce
false positives (spurious dependencies) and false negatives (missed ones).
For high-stakes pipelines, consuming systems should provide an LLM-based
DependencyGraphBuilderProtocol implementation.

Cycle detection: the builder calls DependencyGraph.has_cycle() before
returning and raises DependencyGraphCycleError if one is found.
"""

from __future__ import annotations

import re

from athena.core.exceptions import DependencyGraphCycleError
from athena.core.models.claim import ClaimType
from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import ClaimResult
from athena.core.protocols.dependency import DependencyGraph


class HeuristicDependencyBuilder:
    """
    Default dependency graph builder using type-ordering heuristics.

    No LLM call. No constructor arguments.
    Implements DependencyGraphBuilderProtocol.
    """

    @property
    def builder_id(self) -> str:
        return "HeuristicDependencyBuilder"

    async def build(
        self,
        claim_results: list[ClaimResult],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> DependencyGraph:
        """
        Construct a dependency graph from assessed claim results.

        Raises DependencyGraphCycleError if a cycle is detected.
        """
        graph = DependencyGraph()
        claims = [cr.claim for cr in claim_results]

        # Index by claim_id for fast lookup
        by_id = {c.claim_id: c for c in claims}

        # Partition by type for strategy 1
        factual_ids = [
            c.claim_id for c in claims
            if c.claim_type == ClaimType.factual
        ]
        causal_ids = [
            c.claim_id for c in claims
            if c.claim_type == ClaimType.causal
        ]
        predictive_ids = [
            c.claim_id for c in claims
            if c.claim_type == ClaimType.predictive
        ]
        normative_ids = [
            c.claim_id for c in claims
            if c.claim_type == ClaimType.normative
        ]

        # Strategy 1 + 2: type ordering with scope matching
        for causal_id in causal_ids:
            causal = by_id[causal_id]
            for factual_id in factual_ids:
                factual = by_id[factual_id]
                if _scopes_overlap(causal.scope, factual.scope):
                    graph.add_dependency(causal_id, factual_id)

        for pred_id in predictive_ids:
            pred = by_id[pred_id]
            for factual_id in factual_ids:
                factual = by_id[factual_id]
                if _scopes_overlap(pred.scope, factual.scope):
                    graph.add_dependency(pred_id, factual_id)

        for norm_id in normative_ids:
            norm = by_id[norm_id]
            # Normative claims depend on causal/factual with same scope
            for candidate_id in causal_ids + factual_ids:
                candidate = by_id[candidate_id]
                if _scopes_overlap(norm.scope, candidate.scope):
                    graph.add_dependency(norm_id, candidate_id)

        # Strategy 3: negation detection
        proposition_index = {
            c.claim_id: c.proposition.lower() for c in claims
        }
        for i, claim_a in enumerate(claims):
            for claim_b in claims[i + 1:]:
                if _scopes_overlap(claim_a.scope, claim_b.scope) and _propositions_negate(
                    proposition_index[claim_a.claim_id],
                    proposition_index[claim_b.claim_id],
                ):
                    graph.add_negation(claim_a.claim_id, claim_b.claim_id)

        # Cycle check — compound claims not decomposed at extraction
        if graph.has_cycle():
            raise DependencyGraphCycleError(
                "HeuristicDependencyBuilder detected a cycle in the dependency "
                "graph. This indicates a compound claim was not properly "
                "decomposed at extraction time. "
                f"Builder: {self.builder_id}"
            )

        return graph


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _scope_tokens(scope: str) -> frozenset[str]:
    """Whole-word tokens of the entity/location part (before the first comma)."""
    return frozenset(re.findall(r"\w+", scope.split(",")[0].lower()))


def _scopes_overlap(scope_a: str, scope_b: str) -> bool:
    """
    Do two scope strings refer to the same entity/location?

    Compares the entity part (before the first comma) as sets of WHOLE WORDS:
    overlap when one token set contains the other. Substring matching on raw
    characters made "Corridor A" overlap "Corridor AB" and a one-letter scope
    overlap almost everything. Time references are ignored.
    """
    ta, tb = _scope_tokens(scope_a), _scope_tokens(scope_b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


# (affirmative regex, negative regex). Word-boundary anchored so that
# "should not" is not also matched as "should", and "is high" is not matched
# inside "is higher".
_NEG_PAIRS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = tuple(
    (re.compile(p), re.compile(n))
    for p, n in (
        (r"\bis high\b", r"\bis low\b"),
        (r"\bis increasing\b", r"\bis decreasing\b"),
        (r"\bis present\b", r"\bis absent\b"),
        (r"\bis active\b", r"\bis inactive\b"),
        (r"\bis normal\b", r"\bis abnormal\b"),
        (r"\bwill increase\b", r"\bwill decrease\b"),
        (r"\bshould\b(?! not\b)", r"\bshould not\b"),
    )
)

_STOP = {"the", "a", "an", "is", "are", "was", "were", "it", "this", "that",
         "of", "on", "in", "at"}


def _propositions_negate(prop_a: str, prop_b: str) -> bool:
    """
    Heuristic: do these two propositions contradict each other?

    Detects (1) antonym pairs and (2) an explicit " not " on one side of two
    propositions that share >= 3 content words. Not semantic NLI: the caller
    additionally requires the two claims' scopes to overlap, since opposite
    statements about different entities do not contradict each other.
    """
    for pos, neg in _NEG_PAIRS:
        if (pos.search(prop_a) and neg.search(prop_b)) or (
            neg.search(prop_a) and pos.search(prop_b)
        ):
            return True

    shared = (set(prop_a.split()) & set(prop_b.split())) - _STOP
    if len(shared) >= 3:
        if " not " in prop_a and " not " not in prop_b:
            return True
        if " not " in prop_b and " not " not in prop_a:
            return True
    return False
