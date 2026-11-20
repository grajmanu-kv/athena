"""
util.prompt
===========
Prompt template loader for the Athena library.

Prompts are stored as Markdown files under src/athena/.prompts/<name>/.
Each agent has its own subfolder containing:

    system.md   — static system prompt, loaded once at Agent construction
    message.md  — user message template with $variable substitution,
                  rendered per-call via Prompt.get_text()

Usage
-----
    # Static system prompt (no variables)
    system_text = Prompt("decompose").get_text("system.md")

    # Per-call user message with runtime variables
    user_text = Prompt("decompose").get_text(
        "message.md",
        domain="smart city traffic management",
        risk_level="high",
    )

Template syntax
---------------
Uses stdlib string.Template — $variable or ${variable}.
substitute() raises KeyError if any template variable is not supplied.
This is intentional: missing variables are a programming error, not
a runtime condition to silently ignore.
"""

from __future__ import annotations

import re

from functools import lru_cache
from pathlib import Path
from string import Template

from loguru import logger


class Prompt:
    """
    Loads and renders prompt templates for a named agent.

    Parameters
    ----------
    name:
        Agent name — must match a subfolder under .prompts/.
        e.g. "decompose", "extract"
    """

    # Resolved at class definition time relative to this file:
    # util/prompt.py → util/ → athena/ (root_location)
    root_location: Path = Path(__file__).resolve().parent.parent
    prompt_location: str = ".prompts"
    standard_template: str = "message.md"

    def __init__(self, name: str) -> None:
        self.name = name
        self._prompt_dir = (
            self.root_location / self.prompt_location / self.name
        )

    def get_text(
        self,
        template_name: str = standard_template,
        **kwargs: str,
    ) -> str:
        """
        Load and render a prompt template.

        Parameters
        ----------
        template_name:
            Filename within the agent's prompt folder.
            Defaults to "message.md".
        **kwargs:
            Template variables to substitute.
            All $variables in the template must be supplied.

        Returns
        -------
        str
            Rendered prompt text with all variables substituted.

        Raises
        ------
        FileNotFoundError
            If the template file does not exist.
        KeyError
            If a required template variable is not supplied.
        """
        template_path = self._prompt_dir / template_name

        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}\n"
                f"Expected: src/athena/.prompts/{self.name}/{template_name}"
            )

        raw = template_path.read_text(encoding="utf-8")

        if not kwargs:
            # No substitution needed — return as-is (system.md typical case)
            return raw.strip()

        try:
            return Template(raw).substitute(**kwargs)
        except KeyError as e:
            logger.error(
                "Prompt template variable not supplied: {} "
                "(template={}/{}, required variables not all provided)",
                str(e),
                self.name,
                template_name,
            )
            raise

    @lru_cache(maxsize=8)
    def get_system_prompt(self) -> str:
        """
        Load and cache the static system prompt (system.md).

        Cached per Prompt instance — loaded once per interpreter lifetime.
        Use get_text("system.md") directly if you need uncached access.
        """
        return self.get_text("system.md")


def fence_for(text: str) -> str:
    """
    Return a Markdown code fence strictly longer than any backtick run in
    `text`, so untrusted text embedded between two fences cannot terminate
    the fence early and have its tail read as prompt instructions.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)
