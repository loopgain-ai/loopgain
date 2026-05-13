"""Tests for the opt-in telemetry hook.

Verifies the payload shape (privacy contract — only structural stats),
robust to network failure, and correct integration with LoopGain state.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

import pytest

from loopgain import LoopGain, build_telemetry_payload
from loopgain.telemetry import (
    SCHEMA_VERSION,
    LIBRARY_VERSION,
    build_payload,
    send_payload,
)


def _make_terminated_loop() -> LoopGain:
    """A converged LoopGain instance with a few observations."""
    lg = LoopGain(target_error=0.5, max_iterations=20)
    for e in [10.0, 4.0, 1.5, 0.3]:
        if not lg.should_continue():
            break
        lg.observe(e, output=f"out-{e}")
    return lg


# ----- Payload shape and privacy contract -----


def test_payload_includes_required_fields():
    lg = _make_terminated_loop()
    p = build_payload(lg, workload_id="test-workload")
    assert p["schema_version"] == SCHEMA_VERSION
    assert p["library"] == "loopgain"
    assert p["library_version"] == LIBRARY_VERSION
    assert p["workload_id"] == "test-workload"
    assert "timestamp_hour" in p
    assert "loop" in p
    assert "thresholds" in p
    assert p["smoothing_window"] == 3


def test_payload_loop_section_has_outcome_and_stats():
    lg = _make_terminated_loop()
    p = build_payload(lg)
    loop = p["loop"]
    assert loop["outcome"] == "converged"
    assert loop["iterations_used"] == 4
    assert loop["gain_margin"] is not None
    assert loop["savings_vs_fixed_cap"] is not None
    assert "convergence_profile_summary" in loop
    assert "rollback_triggered" in loop


def test_payload_convergence_profile_summary_only_aggregates():
    """Convergence profile is summarized to min/max/median/samples —
    the raw per-iteration Aβ values are NOT transmitted."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    summary = p["loop"]["convergence_profile_summary"]
    assert set(summary.keys()) == {"min", "max", "median", "samples"}
    # Raw values are not present anywhere in the payload.
    assert "convergence_profile" not in p["loop"]
    assert "error_history" not in p["loop"]


def test_payload_does_not_include_outputs():
    """Best-so-far outputs (which could contain customer content) must
    never appear in the telemetry payload."""
    big_output = {"prompt": "secret customer data", "completion": "more data"}
    lg = LoopGain(target_error=0.5, max_iterations=20)
    for e, out in [(10.0, big_output), (5.0, big_output), (0.3, big_output)]:
        if not lg.should_continue():
            break
        lg.observe(e, output=out)
    p = build_payload(lg)
    p_json = json.dumps(p)
    # Customer data must not leak into the payload, anywhere.
    assert "secret customer data" not in p_json
    assert "prompt" not in p_json
    assert "completion" not in p_json


def test_payload_does_not_include_error_history():
    """Raw error magnitudes are not transmitted — only the summary."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    p_json = json.dumps(p)
    # The raw error_history list is not in the payload.
    assert "error_history" not in p_json


def test_payload_rollback_flag_set_on_divergence():
    """rollback_triggered should be True when the loop diverged."""
    lg = LoopGain(max_iterations=20)
    for e in [10.0, 12.0, 15.0, 20.0, 30.0]:
        if not lg.should_continue():
            break
        lg.observe(e)
    p = build_payload(lg)
    assert p["loop"]["outcome"] in ("diverged", "oscillating")
    assert p["loop"]["rollback_triggered"] is True


def test_payload_rollback_flag_false_on_convergence():
    lg = _make_terminated_loop()
    p = build_payload(lg)
    assert p["loop"]["outcome"] == "converged"
    assert p["loop"]["rollback_triggered"] is False


def test_payload_workload_id_optional():
    """workload_id is optional and may be None."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    assert p["workload_id"] is None


def test_payload_thresholds_included():
    """Threshold values are sent so the receiver knows the config used."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    t = p["thresholds"]
    assert t["fast_converge"] == 0.3
    assert t["converging"] == 0.85
    assert t["stalling"] == 0.95
    assert t["oscillating_upper"] == 1.05


def test_payload_timestamp_hour_bucketed():
    """Timestamp is bucketed to the hour (no minute/second/microsecond)."""
    lg = _make_terminated_loop()
    fixed = datetime(2026, 5, 12, 14, 37, 22, 123456, tzinfo=timezone.utc)
    p = build_payload(lg, timestamp=fixed)
    # The minute/second/microsecond should all be zero in the serialized form.
    assert p["timestamp_hour"].startswith("2026-05-12T14:00:00")
    assert "14:37" not in p["timestamp_hour"]
    assert "22" not in p["timestamp_hour"].split("T")[1]


def test_payload_is_json_serializable():
    """The full payload must round-trip through JSON without errors."""
    lg = _make_terminated_loop()
    p = build_payload(lg, workload_id="rag-v2")
    s = json.dumps(p)
    p2 = json.loads(s)
    assert p2["library_version"] == LIBRARY_VERSION


def test_payload_for_in_progress_loop():
    """Building payload mid-loop is supported (outcome=in_progress)."""
    lg = LoopGain(max_iterations=20)
    lg.observe(10.0)
    lg.observe(5.0)
    p = build_payload(lg)
    assert p["loop"]["outcome"] == "in_progress"


def test_payload_for_not_started_loop():
    """Building payload before any observations doesn't crash."""
    lg = LoopGain()
    p = build_payload(lg)
    assert p["loop"]["outcome"] == "not_started"
    assert p["loop"]["iterations_used"] == 0
    assert p["loop"]["convergence_profile_summary"]["samples"] == 0


# ----- v2 schema: ETA calibration fields -----


def test_payload_schema_version_is_v2():
    """Schema bumped to v2 with the addition of first_eta_* fields."""
    assert SCHEMA_VERSION == 2
    lg = _make_terminated_loop()
    p = build_payload(lg)
    assert p["schema_version"] == 2


def test_payload_includes_first_eta_fields_when_loop_converged():
    """A converging loop produces a captured eta snapshot."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    loop = p["loop"]
    assert "first_eta_prediction" in loop
    assert "first_eta_at_iteration" in loop
    assert loop["first_eta_prediction"] is not None
    assert loop["first_eta_at_iteration"] is not None
    assert loop["first_eta_prediction"] > 0
    assert loop["first_eta_at_iteration"] >= 2


def test_payload_first_eta_none_for_target_zero():
    """target_error=0 means eta is never computable; both fields are None."""
    lg = LoopGain(target_error=0.0, max_iterations=4)
    for _ in range(4):
        lg.observe(10.0)
    p = build_payload(lg)
    assert p["loop"]["first_eta_prediction"] is None
    assert p["loop"]["first_eta_at_iteration"] is None


def test_payload_first_eta_none_for_diverging_loop():
    """A divergent loop never produces a positive eta."""
    lg = LoopGain(target_error=0.5, max_iterations=20)
    for e in [10.0, 12.0, 15.0, 20.0, 30.0]:
        if not lg.should_continue():
            break
        lg.observe(e)
    p = build_payload(lg)
    assert p["loop"]["first_eta_prediction"] is None
    assert p["loop"]["first_eta_at_iteration"] is None


# ----- send_payload behavior -----


def test_send_payload_returns_false_on_unreachable_endpoint():
    """A network failure must NOT raise — telemetry is best-effort."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    # Use a known-bad endpoint; should fail quickly and return False.
    ok = send_payload(
        "http://127.0.0.1:1/v1/aggregate",  # nothing listens on port 1
        token="fake",
        payload=p,
        timeout=0.5,
    )
    assert ok is False


def test_send_telemetry_method_on_loopgain_returns_false_on_failure():
    """LoopGain.send_telemetry is best-effort: returns False on failure."""
    lg = _make_terminated_loop()
    ok = lg.send_telemetry(
        endpoint="http://127.0.0.1:1/v1/aggregate",
        token="fake",
        workload_id="test",
        timeout=0.5,
    )
    assert ok is False


def test_send_telemetry_does_not_raise_on_bad_url():
    """Even a syntactically-invalid URL should be swallowed."""
    lg = _make_terminated_loop()
    # Should not raise; just return False.
    result = lg.send_telemetry(
        endpoint="not-a-real-url",
        token="fake",
        timeout=0.5,
    )
    assert result is False


# ----- Reachable but rejecting endpoint via mock -----


def test_send_payload_constructs_correct_request(monkeypatch):
    """Verify the POST is constructed with the right headers and body."""
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    lg = _make_terminated_loop()
    p = build_payload(lg, workload_id="test")
    ok = send_payload(
        "https://telemetry.loopgain.ai/v1/aggregate",
        token="my-token",
        payload=p,
        timeout=1.5,
    )
    assert ok is True
    assert captured["url"] == "https://telemetry.loopgain.ai/v1/aggregate"
    assert captured["method"] == "POST"
    # Headers (case-insensitive in HTTP, but urllib.Request normalizes to titlecase).
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["content-type"] == "application/json"
    assert headers["authorization"] == "Bearer my-token"
    assert "loopgain/" in headers["user-agent"]
    # Body is the JSON-encoded payload.
    body = json.loads(captured["body"])
    assert body["library_version"] == LIBRARY_VERSION
    assert captured["timeout"] == 1.5
