"""
athena.web.settings
===================
Runtime configuration, read from the environment (and an optional .env file).

    GEMINI_API_KEY | GOOGLE_API_KEY   Gemini API key (server side only; never
                                      sent to or entered in the browser)
    ATHENA_MODEL                      Gemini model id (default: gemini-2.5-flash)
    ATHENA_TIMEOUT_S                  Whole-run timeout in seconds (default 180)
    ATHENA_MAX_CHARS                  Max characters per text field (default 20000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op when there is no .env file

DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    timeout_s: float
    max_chars: int

    @property
    def key_configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
            model=os.getenv("ATHENA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout_s=float(os.getenv("ATHENA_TIMEOUT_S", "180")),
            max_chars=int(os.getenv("ATHENA_MAX_CHARS", "20000")),
        )
