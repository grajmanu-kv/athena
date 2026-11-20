"""
util.ids
========
UUID4 generation for Athena pipeline artifacts.

Centralised here so tests can monkeypatch a single location
to produce deterministic IDs without patching uuid.uuid4 globally.
"""

from __future__ import annotations

import uuid


def new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())