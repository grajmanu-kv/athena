"""
tests/test_web_api.py
=====================
End-to-end tests of the FastAPI harness. The real Athena pipeline runs; only the
LLM is replaced by a scripted FunctionModel (see _scripted.Script).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from tests._scripted import Script, good_claims
from athena.web.app import create_app
from athena.web.settings import Settings

CFG = Settings(api_key=None, model="gemini-test", timeout_s=30, max_chars=2000)


class WebScript(Script):
    """Script that also answers the KnownFactsChecker's output schema."""

    def __init__(self, claims, relation="silent", fact_number=None, **kw):
        super().__init__(claims, **kw)
        self.relation, self.fact_number = relation, fact_number

    def __call__(self, messages, info):
        tool = info.output_tools[0]
        if "relation" in tool.parameters_json_schema.get("properties", {}):
            self.calls.append("facts")
            args = {"relation": self.relation, "reasoning": "scripted"}
            if self.fact_number is not None:
                args["fact_number"] = self.fact_number
            return ModelResponse(parts=[ToolCallPart(tool.name, args)])
        return super().__call__(messages, info)


def client(script) -> TestClient:
    return TestClient(create_app(CFG, model_factory=lambda name: FunctionModel(script)))


BODY = dict(
    query="You are a traffic advisor. Report sensor readings. Do not infer causation.",
    response=(
        "Sensor CAM-042 reports 87 veh/km on Corridor A. "
        "Density will likely exceed 100 veh/km within 15 minutes. "
        "The operator should notify the duty officer."
    ),
    goal="Brief the duty officer",
    action_class="recommend",
    consequence_class="low",
    domain="traffic",
)


def test_health_and_index():
    c = client(WebScript(good_claims()))
    assert c.get("/api/health").json()["status"] == "ok"
    assert "Athena" in c.get("/").text


def test_risk_endpoint_matches_library_matrix():
    c = client(WebScript(good_claims()))
    assert c.get("/api/risk", params=dict(action_class="automate", consequence_class="critical")).json() == {"risk_level": "critical"}
    assert c.get("/api/risk", params=dict(action_class="inform", consequence_class="low")).json() == {"risk_level": "low"}


def test_validate_without_facts_tops_out_at_plausible():
    r = client(WebScript(good_claims())).post("/api/validate", json=BODY)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total_claims"] == 3
    assert d["counts"]["verified"] == 0
    assert d["verdict"]["code"] in {"accepted", "accepted_with_warnings"}
    assert any("known_facts" in n["text"] for n in d["notes"])
    assert d["risk_level_derived"] == d["verdict"]["risk_level"] == "medium"


def test_known_fact_entailment_marks_factual_claim_verified():
    body = {**BODY, "known_facts": ["Sensor CAM-042 reported 87 veh/km on Corridor A"]}
    d = client(WebScript(good_claims(), relation="entails", fact_number=1)).post("/api/validate", json=body).json()
    by_type = {c["type"]: c for c in d["claims"]}
    assert by_type["factual"]["state"] == "verified"
    assert "fact #1" in by_type["factual"]["rationale"]
    # non-factual claims are never promoted by the facts checker
    assert by_type["predictive"]["state"] != "verified"
    assert by_type["normative"]["state"] != "verified"


def test_known_fact_contradiction_rejects_at_all_levels():
    body = {**BODY, "known_facts": ["Corridor A density is 12 veh/km"]}
    d = client(WebScript(good_claims(), relation="contradicts", fact_number=1)).post("/api/validate", json=body).json()
    assert d["verdict"]["code"] == "rejected"
    assert any(c["state"] == "contradicted" for c in d["claims"])


def test_entails_without_valid_citation_is_not_promoted():
    body = {**BODY, "known_facts": ["one fact"]}
    d = client(WebScript(good_claims(), relation="entails", fact_number=9)).post("/api/validate", json=body).json()
    assert d["counts"]["verified"] == 0


def test_guard_outage_is_reported_not_hidden():
    body = {**BODY, "constraints": ["Must cite a sensor ID"]}
    d = client(WebScript(good_claims(), fail_guards=True)).post("/api/validate", json=body).json()
    assert any("could not run" in n["text"] for n in d["notes"])
    outcomes = {g["outcome"] for c in d["claims"] for g in c["guards"]}
    assert "errored" in outcomes and "failed" not in outcomes


def test_critical_risk_is_not_auto_accepted():
    body = {**BODY, "action_class": "automate", "consequence_class": "critical"}
    d = client(WebScript(good_claims())).post("/api/validate", json=body).json()
    assert d["verdict"]["risk_level"] == "critical"
    assert d["verdict"]["code"] in {"rejected", "escalated"}


def test_missing_key_gives_clear_503():
    c = TestClient(create_app(CFG))  # real factory, no key configured
    r = c.post("/api/validate", json=BODY)
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["error"]


def test_oversize_and_bad_input_are_rejected_cleanly():
    c = client(WebScript(good_claims()))
    assert c.post("/api/validate", json={**BODY, "response": "x" * 5000}).status_code == 413
    r = c.post("/api/validate", json={**BODY, "action_class": "nope"})
    assert r.status_code == 422 and "action_class" in r.json()["error"]
    assert c.post("/api/validate", json={**BODY, "query": "   "}).status_code == 422


def test_provider_failure_maps_to_502():
    def boom(messages, info):
        raise RuntimeError("provider down")
    boom.__name__ = "boom"
    d = client(boom).post("/api/validate", json=BODY)
    assert d.status_code == 502 and "error" in d.json()
