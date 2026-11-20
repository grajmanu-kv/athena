"""
athena.web.llm
==============
Builds the pydantic-ai Gemini model that Athena's stages (decompose, extract,
three guards) and the known-facts checker run on.
"""

from __future__ import annotations

from pydantic_ai.models import Model

from athena.web.settings import Settings


class ModelNotConfigured(RuntimeError):
    """Raised when no Gemini API key is present in the server environment."""


def build_gemini_model(settings: Settings, model_name: str | None = None) -> Model:
    if not settings.api_key:
        raise ModelNotConfigured(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in "
            "the server environment or in a .env file, then restart."
        )
    # Imported here so the app (and its tests) import without google-genai.
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    name = (model_name or settings.model).removeprefix("google-gla:").removeprefix("gemini:")
    return GoogleModel(name, provider=GoogleProvider(api_key=settings.api_key))
