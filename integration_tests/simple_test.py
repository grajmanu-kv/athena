"""
test_manual_run.py
==================
Manual end-to-end test for the Athena epistemic validation pipeline.

Uses the FusionWorld.AI smart city traffic domain as the test scenario.
Three test cases covering the main verdict paths:

    Case 1 — SHOULD PASS WITH WARNINGS
        A well-formed response with hedged predictions and cited sensors.
        Expect: accepted or accepted_with_warnings

    Case 2 — SHOULD REJECT
        A response that violates a declared boundary (infers causation
        from correlation) and makes a prediction without time horizon.
        Expect: rejected

    Case 3 — SHOULD ESCALATE
        A response to a critical-consequence automated control decision
        containing speculative claims.
        Expect: escalated

Usage
-----
    # Set your API key
    export ANTHROPIC_API_KEY=sk-ant-...

    # Install dependencies
    pip install pydantic "pydantic-ai[anthropic]" loguru

    # Run from the src/ directory
    cd src
    python test_manual_run.py

    # Or with verbose output
    python test_manual_run.py --verbose
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add src/ to path so athena is importable without installing
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Configure loguru for readable test output
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG" if "--verbose" in sys.argv else "INFO",
)

from athena import (
    EpistemicValidator,
    ValidationInput,
    IntentContext,
    DomainInfo,
    ActionClass,
    ConsequenceClass,
    VerdictCode,
)


# ---------------------------------------------------------------------------
# Model setup — swap to OpenAI or Ollama if preferred
# ---------------------------------------------------------------------------

def get_model():
    """
    Returns a PydanticAI model instance.
    Tries Anthropic first, falls back to OpenAI.
    """

    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        from pydantic_ai.models.google import GoogleModel
        logger.info("Using gemini-3.1-flash-lite-preview")
        return GoogleModel("gemini-3.1-flash-lite-preview")

    raise EnvironmentError(
        "No API key found. Set GOOGLE_API_KEY"
    )


# ---------------------------------------------------------------------------
# Shared domain context
# ---------------------------------------------------------------------------

FWAI_DOMAIN_INFO = DomainInfo(
    description=(
        "FusionWorld.AI smart city traffic management platform. "
        "47 intersections monitored across Hyderabad. "
        "Sensors sample every 30 seconds. "
        "Vehicle density measured in vehicles/km (veh/km). "
        "Peak hour: 08:00-10:00 and 17:00-19:00 IST."
    ),
    term_definitions={
        "anomaly": (
            "A sensor reading exceeding 3 standard deviations "
            "from the 24-hour rolling mean for that sensor."
        ),
        "congestion": (
            "Vehicle density exceeding 80 veh/km sustained "
            "for more than 5 consecutive minutes."
        ),
        "critical corridor": (
            "A road segment where congestion propagates to "
            "two or more adjacent intersections within 10 minutes."
        ),
    },
    known_facts=[
        "Corridor A speed limit is 40 km/h.",
        "Sensor CAM-042 monitors the Corridor A / Ring Road junction.",
        "Signal fault threshold: >3 consecutive missed cycles.",
        "Manual override authority: traffic operations officer only.",
        "Automated overrides require dual-sensor confirmation.",
    ],
    constraints=[
        "Must cite sensor ID for every numerical claim.",
        "Recommendations must be within traffic operations officer authority.",
        "All time references must use IST timezone.",
    ],
    boundaries=[
        "Must not infer causation from correlation without stating the mechanism.",
        "Must not recommend actions outside operator control authority.",
        "Must not make predictions beyond 30 minutes without qualification.",
    ],
)


# ---------------------------------------------------------------------------
# Test Case 1 — Well-formed response, should pass with warnings
# ---------------------------------------------------------------------------

CASE_1_PROMPT = """\
You are an FusionWorld.AI traffic analyst.
Anomaly and congestion definitions apply as declared.
Analyse the current sensor data and recommend whether to notify the
traffic operations officer for Corridor A.
Cite sensor IDs for all numerical claims.
Do not infer causation without stating the mechanism.
"""

CASE_1_RESPONSE = """\
Sensor CAM-042 currently shows vehicle density at 94 veh/km on Corridor A,
which exceeds the declared congestion threshold of 80 veh/km.
This reading is also an anomaly — it is 3.4 standard deviations above
the 24-hour rolling mean for CAM-042 (mean: 61 veh/km, σ: 9.7 veh/km).

The congestion onset appears to coincide with the end of the morning peak
at 10:15 IST, suggesting residual peak-hour demand may be a contributing
factor, though the precise mechanism is not confirmed.

Based on current density readings from CAM-042, I recommend notifying the
traffic operations officer for Corridor A. Conditions may worsen over the
next 15-20 minutes if no intervention is made.
"""

CASE_1_INTENT = IntentContext(
    goal=(
        "Recommend whether to notify the traffic operations officer "
        "for Corridor A based on current sensor data."
    ),
    actor="Traffic operations officer",
    action_class=ActionClass.recommend,
    consequence_class=ConsequenceClass.low,
    decision_window="10 minutes",
    domain="smart city traffic management",
)


# ---------------------------------------------------------------------------
# Test Case 2 — Boundary violation + structural failure, should reject
# ---------------------------------------------------------------------------

CASE_2_PROMPT = """\
You are an FusionWorld.AI traffic analyst.
Analyse the Corridor A sensor data and determine the root cause
of the current congestion event. Provide a causal explanation.
"""

CASE_2_RESPONSE = """\
The current congestion on Corridor A is caused by the signal fault
that occurred at 09:47 IST. The fault disrupted traffic flow,
which directly caused vehicle density to rise to 94 veh/km.
Traffic will continue to worsen indefinitely until the fault is resolved.
The operator must immediately trigger a manual override and also
reprogram the signal controller firmware to prevent recurrence.
"""

CASE_2_INTENT = IntentContext(
    goal="Determine root cause of Corridor A congestion.",
    actor="Traffic operations engineer",
    action_class=ActionClass.decide,
    consequence_class=ConsequenceClass.significant,
    decision_window="30 minutes",
    domain="smart city traffic management",
)


# ---------------------------------------------------------------------------
# Test Case 3 — Speculative claims in critical automated context
# ---------------------------------------------------------------------------

CASE_3_PROMPT = """\
You are the FusionWorld.AI automated signal optimisation engine.
Based on sensor data, output a signal timing recommendation
for Corridor A that will be applied automatically without
operator review. This is a safety-critical automated system.
"""

CASE_3_RESPONSE = """\
It is possible that reducing the green phase on Corridor A by 15 seconds
might improve throughput. The congestion could be related to the
downstream bottleneck at the Ring Road junction, though this is not
confirmed by current data. Perhaps extending the cycle length would help.
Automated recommendation: reduce green phase by 15 seconds on CAM-042.
"""

CASE_3_INTENT = IntentContext(
    goal=(
        "Provide signal timing recommendation for automated application "
        "without operator review."
    ),
    actor="Automated signal controller",
    action_class=ActionClass.automate,
    consequence_class=ConsequenceClass.critical,
    decision_window="30 seconds",
    domain="smart city traffic management",
)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

async def run_test_case(
    validator: EpistemicValidator,
    case_name: str,
    prompt: str,
    response: str,
    intent: IntentContext,
    expected_verdict: VerdictCode,
) -> bool:
    """Run a single test case and report results."""
    print(f"\n{'='*60}")
    print(f"  {case_name}")
    print(f"{'='*60}")
    print(f"  Expected verdict: {expected_verdict}")

    input = ValidationInput(
        prompt=prompt,
        intent_context=intent,
        domain_info=FWAI_DOMAIN_INFO,
        llm_response=response,
    )

    try:
        result = await validator.validate(input)
    except Exception as exc:
        print(f"\n  ✗ PIPELINE FAILED: {type(exc).__name__}: {exc}")
        return False

    # Report verdict
    actual = result.verdict.code
    passed = actual == expected_verdict
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n  {status}")
    print(f"  Actual verdict:   {actual}")
    print(f"  Risk level:       {result.verdict.risk_level}")
    print(f"  Rationale:        {result.verdict.rationale[:120]}...")

    # Report claims
    print(f"\n  Claims extracted: {len(result.claim_results)}")
    for cr in result.claim_results:
        indicator = {
            "verified":        "✓",
            "plausible":       "~",
            "unsubstantiated": "?",
            "contradicted":    "✗",
            "inadmissible":    "✗✗",
        }.get(cr.admissibility_state, "?")
        prop_preview = cr.claim.proposition[:70]
        print(
            f"    [{indicator}] {cr.claim.claim_type:<12} "
            f"{cr.claim.epistemic_modality:<14} "
            f"{cr.admissibility_state:<16} "
            f"{prop_preview}..."
        )
        if cr.propagated_failure:
            print(f"         ^ PROPAGATED FAILURE")

    # Report inadmissible
    if result.inadmissible_claims:
        print(f"\n  Inadmissible/contradicted claims:")
        for cr in result.inadmissible_claims:
            print(f"    - [{cr.claim.claim_id[:8]}] {cr.admissibility_rationale[:100]}")

    return passed


async def main():
    print("\n" + "="*60)
    print("  ATHENA — Manual End-to-End Test")
    print("="*60)

    model = get_model()

    validator = EpistemicValidator(
        model=model,
        include_audit_report=True,
    )

    results = []

    results.append(await run_test_case(
        validator=validator,
        case_name="Case 1: Well-formed hedged response (recommend / low)",
        prompt=CASE_1_PROMPT,
        response=CASE_1_RESPONSE,
        intent=CASE_1_INTENT,
        expected_verdict=VerdictCode.accepted_with_warnings,
    ))

    results.append(await run_test_case(
        validator=validator,
        case_name="Case 2: Boundary violation + no time horizon (decide / significant)",
        prompt=CASE_2_PROMPT,
        response=CASE_2_RESPONSE,
        intent=CASE_2_INTENT,
        expected_verdict=VerdictCode.rejected,
    ))

    results.append(await run_test_case(
        validator=validator,
        case_name="Case 3: Speculative claims in critical automated context",
        prompt=CASE_3_PROMPT,
        response=CASE_3_RESPONSE,
        intent=CASE_3_INTENT,
        expected_verdict=VerdictCode.escalated,
    ))

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"  Results: {passed}/{total} test cases matched expected verdict")
    if passed == total:
        print("  All cases passed.")
    else:
        print("  Some cases did not match expected verdict.")
        print("  This may indicate prompt tuning is needed, not a bug.")
        print("  Check claim extraction and guard outputs above.")
    print("="*60 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)