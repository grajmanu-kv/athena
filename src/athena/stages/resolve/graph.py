"""
stages.resolve.graph
====================
Re-exports DependencyGraph from core.protocols.dependency.

DependencyGraph is defined in core (zero internal dependencies) because
it is part of the DependencyGraphBuilderProtocol contract — consuming
systems that implement the protocol must be able to reference it without
importing from stages/.

This module provides a stage-local import path for pipeline stages that
need DependencyGraph without reaching into core.protocols directly.
"""

from athena.core.protocols.dependency import DependencyGraph

__all__ = ["DependencyGraph"]