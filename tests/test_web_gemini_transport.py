"""
tests/test_web_gemini_transport.py
==================================
Offline check that the real pipeline works through pydantic-ai's GoogleModel.

Only the google-genai network call is stubbed. Everything above it is real:
Athena's stage schemas -> pydantic-ai tool definitions -> Gemini request config,
and a Gemini `function_call` reply -> parsed stage output. This catches schema
shapes the Gemini request builder rejects. It does NOT prove the live Gemini
service accepts them (no network, no key); see the README section.
"""

from __future__ import annotations

import asyncio

from google.genai import types as T
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from athena import EpistemicValidator
from athena.web.known_facts import KnownFactsChecker
from tests._scripted import good_claims, make_input


def _reply(name: str, args: dict) -> T.GenerateContentResponse:
    return T.GenerateContentResponse(
        candidates=[T.Candidate(
            content=T.Content(role="model", parts=[T.Part(function_call=T.FunctionCall(name=name, args=args))]),
            finish_reason=T.FinishReason.STOP)],
        usage_metadata=T.GenerateContentResponseUsageMetadata(
            prompt_token_count=1, candidates_token_count=1, total_token_count=2),
        model_version="stub",
    )


def test_pipeline_runs_through_google_model_with_stubbed_transport():
    seen: set[str] = set()

    async def fake_generate(*, model, contents, config):
        cfg = config if isinstance(config, dict) else config.model_dump()
        tool = cfg["tools"][0]
        decl = (tool["function_declarations"] if isinstance(tool, dict) else tool.function_declarations)[0]
        get = decl.get if isinstance(decl, dict) else (lambda k: getattr(decl, k, None))
        schema = get("parameters_json_schema") or get("parameters")
        schema = schema.model_dump(exclude_none=True) if hasattr(schema, "model_dump") else schema
        props = set(schema.get("properties", {}))
        if "response" in props:
            seen.add("extract"); args = {"response": good_claims()}
        elif "implicit_constraints" in props:
            seen.add("decompose")
            args = {"implicit_constraints": [], "implicit_boundaries": [], "term_definitions": {},
                    "undeclared_terms": [], "injected_facts": []}
        elif "relation" in props:
            seen.add("facts"); args = {"relation": "entails", "fact_number": 1, "reasoning": "stub"}
        else:
            seen.add("guard")
            args = ({"satisfies": True} if "satisfies" in props
                    else {"violates": False} if "violates" in props else {"consistent": True})
            args["reasoning"] = "ok"
        return _reply(get("name"), args)

    async def run():
        model = GoogleModel("gemini-2.5-flash", provider=GoogleProvider(api_key="fake-key"))
        model.client.aio.models.generate_content = fake_generate
        validator = EpistemicValidator(
            model=model,
            admissibility_checkers=[KnownFactsChecker(model, ["Sensor CAM-042 reported 87 veh/km on Corridor A"])],
            include_audit_report=True,
        )
        return await validator.validate(make_input())

    result = asyncio.run(run())
    assert {"decompose", "extract", "facts", "guard"} <= seen
    assert [str(c.admissibility_state) for c in result.claim_results] == ["verified", "plausible", "plausible"]
    assert result.verdict.code == "accepted"
