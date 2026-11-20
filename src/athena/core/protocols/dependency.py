"""
core.protocols.dependency
==========================
Protocol definition for dependency graph builders.

The dependency graph builder (Step 5) receives all assessed ClaimResults
and constructs a directed graph where:
    nodes = claims
    edges = epistemic dependencies (depends_on) and contradictions (negates)

Failure propagation runs on this graph: if an upstream claim is inadmissible
or contradicted, downstream claims that depend on it are downgraded.

The SDK ships a default HeuristicDependencyBuilder that infers dependencies
from claim type ordering rules and proposition text similarity. Consuming
systems may replace it with a more precise builder — including an LLM-based
builder for complex reasoning chains.

Design contract
---------------
* The builder receives ClaimResults (claims after guard + admissibility
  evaluation), not raw Claims. It has access to admissibility_state
  to make topology decisions.

* The builder returns a DependencyGraph — a lightweight structure that
  the FailurePropagator (stages/resolve/propagator.py) traverses.

* The builder MUST NOT modify ClaimResult objects. It only constructs
  the graph topology. Propagation is the Propagator's responsibility.

* Cycles in the dependency graph are a structural error and must be
  detected and reported by the builder — not silently ignored.
  A cyclic dependency graph indicates a compound claim was not
  properly decomposed at extraction time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from athena.core.models.decomposed_prompt import DecomposedPrompt
from athena.core.models.input import IntentContext
from athena.core.models.result import ClaimResult


@runtime_checkable
class DependencyGraphBuilderProtocol(Protocol):
    """
    Protocol for dependency graph builders.

    Example implementation (LLM-assisted builder):

        class LLMDependencyBuilder:
            def __init__(self, model: Model) -> None:
                self._model = model

            @property
            def builder_id(self) -> str:
                return "LLMDependencyBuilder"

            async def build(
                self,
                claim_results: list[ClaimResult],
                decomposed_prompt: DecomposedPrompt,
                intent_context: IntentContext,
            ) -> DependencyGraph:
                # call LLM to identify dependency edges
                # construct and return DependencyGraph
                ...
    """

    @property
    def builder_id(self) -> str:
        """
        Stable string identifier for this builder.
        Convention: use the class name.
        """
        ...

    async def build(
        self,
        claim_results: list[ClaimResult],
        decomposed_prompt: DecomposedPrompt,
        intent_context: IntentContext,
    ) -> "DependencyGraph":
        """
        Construct a dependency graph from assessed claim results.

        Parameters
        ----------
        claim_results:
            All ClaimResults after guard and admissibility evaluation.
            Do not mutate.
        decomposed_prompt:
            Structured prompt decomposition. Available for context
            when inferring dependencies from claim propositions.
        intent_context:
            The declared intent. Available for domain-aware
            dependency inference.

        Returns
        -------
        DependencyGraph
            The constructed graph. Must be acyclic.
            Raises DependencyGraphCycleError if a cycle is detected.
        """
        ...


class DependencyGraph:
    """
    Directed graph of epistemic dependencies between claims.

    Nodes are claim_ids. Edges are directed dependencies.

    Two edge types:
        depends_on  : source claim presupposes target claim is valid.
                      If target is inadmissible, source is downgraded.
        negates     : source claim logically contradicts target claim.
                      Both are flagged; the higher-confidence claim
                      (by epistemic_modality) is tentatively preferred.

    This is a plain Python class — not a Pydantic model — because
    graph topology is mutable during construction by the builder.
    It becomes logically immutable once returned to the pipeline.
    """

    __slots__ = ("_depends_on", "_negates")

    def __init__(self) -> None:
        # claim_id → set of claim_ids it depends on
        self._depends_on: dict[str, set[str]] = {}
        # claim_id → set of claim_ids it negates
        self._negates: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Mutation (builder phase only)
    # ------------------------------------------------------------------

    def add_dependency(self, from_claim_id: str, to_claim_id: str) -> None:
        """
        Declare that from_claim_id epistemically depends on to_claim_id.
        i.e. from_claim presupposes to_claim is valid.
        """
        if from_claim_id not in self._depends_on:
            self._depends_on[from_claim_id] = set()
        self._depends_on[from_claim_id].add(to_claim_id)

    def add_negation(self, claim_id_a: str, claim_id_b: str) -> None:
        """
        Declare that claim_id_a and claim_id_b logically contradict.
        Negation is symmetric — both directions are recorded.
        """
        if claim_id_a not in self._negates:
            self._negates[claim_id_a] = set()
        self._negates[claim_id_a].add(claim_id_b)

        if claim_id_b not in self._negates:
            self._negates[claim_id_b] = set()
        self._negates[claim_id_b].add(claim_id_a)

    # ------------------------------------------------------------------
    # Traversal (propagation phase)
    # ------------------------------------------------------------------

    def dependencies_of(self, claim_id: str) -> frozenset[str]:
        """
        Returns the set of claim_ids that claim_id depends on.
        Empty frozenset if the claim has no declared dependencies.
        """
        return frozenset(self._depends_on.get(claim_id, set()))

    def negations_of(self, claim_id: str) -> frozenset[str]:
        """
        Returns the set of claim_ids that claim_id negates (or is negated by).
        Empty frozenset if no negations declared.
        """
        return frozenset(self._negates.get(claim_id, set()))

    def all_claim_ids(self) -> frozenset[str]:
        """All claim_ids that appear in any edge."""
        ids: set[str] = set()
        ids.update(self._depends_on.keys())
        for targets in self._depends_on.values():
            ids.update(targets)
        ids.update(self._negates.keys())
        return frozenset(ids)

    def has_cycle(self) -> bool:
        """
        Detect cycles in the depends_on graph using iterative DFS.
        Negation edges are not checked for cycles — mutual negation
        between two claims is valid.
        Returns True if any cycle exists.
        """
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbour in self._depends_on.get(node, set()):
                if neighbour not in visited:
                    if dfs(neighbour):
                        return True
                elif neighbour in in_stack:
                    return True
            in_stack.discard(node)
            return False

        for claim_id in list(self._depends_on.keys()):
            if claim_id not in visited:
                if dfs(claim_id):
                    return True
        return False

    def topological_order(self) -> list[str]:
        """
        Return claim_ids in topological order (dependencies before dependents).
        Used by FailurePropagator to process claims in correct order.
        Raises DependencyGraphCycleError if a cycle exists.
        """
        from athena.core.exceptions import DependencyGraphCycleError

        if self.has_cycle():
            raise DependencyGraphCycleError(
                "Dependency graph contains a cycle. This indicates a compound "
                "claim was not properly decomposed at extraction time. "
                "Review the extraction prompt or split the offending claims."
            )

        all_ids = self.all_claim_ids()
        in_degree: dict[str, int] = {cid: 0 for cid in all_ids}
        for cid in all_ids:
            for dep in self._depends_on.get(cid, set()):
                if dep in in_degree:
                    in_degree[cid] = in_degree.get(cid, 0)
                # dep is pointed to by cid — in_degree of dep is not
                # incremented; in_degree tracks how many nodes point TO cid
        # rebuild correctly: in_degree[node] = number of nodes that depend on node
        in_degree = {cid: 0 for cid in all_ids}
        for cid, deps in self._depends_on.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 0  # dep is a prerequisite, not a dependent
                # cid depends on dep → cid has in_degree contribution from dep
        # Standard Kahn's algorithm on reversed edges
        # Edge: cid → dep means "cid depends on dep"
        # For topological order we need deps before dependents
        # Build reverse: dep → [cid, ...] (dep must come before cid)
        reverse: dict[str, list[str]] = {cid: [] for cid in all_ids}
        in_deg: dict[str, int] = {cid: 0 for cid in all_ids}
        for cid, deps in self._depends_on.items():
            for dep in deps:
                if dep not in reverse:
                    reverse[dep] = []
                reverse[dep].append(cid)
                in_deg[cid] = in_deg.get(cid, 0) + 1

        queue = [cid for cid, deg in in_deg.items() if deg == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dependent in reverse.get(node, []):
                in_deg[dependent] -= 1
                if in_deg[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(all_ids):
            raise DependencyGraphCycleError(
                "Topological sort incomplete — cycle detected."
            )
        return order

    def __repr__(self) -> str:
        return (
            f"DependencyGraph("
            f"nodes={len(self.all_claim_ids())}, "
            f"dependency_edges={sum(len(v) for v in self._depends_on.values())}, "
            f"negation_pairs={sum(len(v) for v in self._negates.values()) // 2})"
        )