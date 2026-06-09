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
    assert loop["savings_vs_fixed_cap"] is not None
    assert "convergence_profile_summary" in loop
    assert "rollback_triggered" in loop


def test_payload_convergence_profile_summary_present():
    """Convergence profile summary is min/max/median/samples on loop.*."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    summary = p["loop"]["convergence_profile_summary"]
    assert set(summary.keys()) == {"min", "max", "median", "samples"}


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


def test_payload_omits_per_iteration_when_disabled():
    """include_per_iteration=False sends only aggregate summary stats."""
    lg = _make_terminated_loop()
    p = build_payload(lg, include_per_iteration=False)
    assert "per_iteration" not in p
    # The summary on loop.* is unchanged.
    assert "convergence_profile_summary" in p["loop"]


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


def test_payload_serializes_strict_json_for_constant_error_trajectory():
    """A zero-error trajectory pushes Aβ to +inf (E(n)/E(n-1) with E(n-1)=0).

    Standard JSON forbids Infinity / NaN, and the receiver rejects payloads
    that include them. The build_payload sanitizer must coerce non-finite
    floats to None so the payload still round-trips through a strict parser.
    """
    lg = LoopGain(max_iterations=5)
    for _ in range(5):
        if not lg.should_continue():
            break
        lg.observe(0.0, output="x")
    p = build_payload(lg)
    # Strict round-trip: allow_nan=False raises on inf/nan.
    encoded = json.dumps(p, allow_nan=False)
    decoded = json.loads(encoded)
    # Per-iteration Aβ values can be non-finite (E(n)/E(n-1) with E(n-1)=0)
    # and the convergence-profile summary must stay finite-or-None too.
    for v in decoded["per_iteration"]["convergence_profile"]:
        assert v is None or isinstance(v, (int, float))


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


# ----- schema version -----


def test_payload_schema_version_is_v4():
    """Schema bumped to v4 when ETA + gain_margin were removed from the payload."""
    assert SCHEMA_VERSION == 4
    lg = _make_terminated_loop()
    p = build_payload(lg)
    assert p["schema_version"] == 4


def test_payload_loop_section_drops_eta_and_gain_margin():
    """v4 no longer carries the discontinued ETA / gain_margin fields."""
    lg = _make_terminated_loop()
    loop = build_payload(lg)["loop"]
    assert "gain_margin" not in loop
    assert "first_eta_prediction" not in loop
    assert "first_eta_at_iteration" not in loop


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

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

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


# ----- Scheme allow-list (0.1.5) -----
#
# The library refuses to attach the bearer token to anything but
# `https://` by default. `http://` is allowed only when the caller
# explicitly opts in with `allow_insecure=True` (intended for local dev).
# Every other scheme (`file://`, `javascript:`, `ftp://`, ...) is rejected
# unconditionally so a misconfigured or coerced endpoint cannot exfiltrate
# the token via an unintended channel.


def test_send_payload_rejects_http_by_default(monkeypatch):
    """http:// endpoints are rejected without ever calling urlopen."""
    called = {"n": 0}

    def fake_urlopen(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("urlopen must not be called for rejected scheme")

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    p = build_payload(lg)
    ok = send_payload(
        "http://telemetry.loopgain.ai/v1/aggregate",
        token="my-token",
        payload=p,
        timeout=1.5,
    )
    assert ok is False
    assert called["n"] == 0


def test_send_payload_allows_http_with_allow_insecure_true(monkeypatch):
    """http:// is permitted when allow_insecure=True (local-dev escape hatch)."""

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return FakeResponse()

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    p = build_payload(lg)
    ok = send_payload(
        "http://localhost:8787/v1/aggregate",
        token="my-token",
        payload=p,
        timeout=1.5,
        allow_insecure=True,
    )
    assert ok is True
    assert captured["url"] == "http://localhost:8787/v1/aggregate"


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com/foo",
        "data:text/plain,hello",
        "gopher://example.com/",
    ],
)
def test_send_payload_rejects_exotic_schemes(monkeypatch, endpoint):
    """Non-http(s) schemes are rejected even when allow_insecure=True —
    the bearer token must never leave via an unintended channel."""

    def fake_urlopen(*args, **kwargs):
        raise AssertionError("urlopen must not be called for rejected scheme")

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    p = build_payload(lg)
    # Neither default nor allow_insecure=True should permit exotic schemes.
    assert send_payload(endpoint, token="t", payload=p) is False
    assert send_payload(endpoint, token="t", payload=p, allow_insecure=True) is False


def test_send_payload_https_still_works_after_scheme_check(monkeypatch):
    """The canonical https://telemetry.loopgain.ai path is unchanged."""

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, timeout=None):
        return FakeResponse()

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    p = build_payload(lg)
    ok = send_payload(
        "https://telemetry.loopgain.ai/v1/aggregate",
        token="my-token",
        payload=p,
    )
    assert ok is True


# ----- v3 schema: per-iteration trajectories + classification fields -----


def test_payload_includes_per_iteration_by_default():
    """Per-iteration trajectories are included by default and contain
    one error entry per observe() call. The Aβ trajectory is shorter
    (no Aβ for the first observation, plus a TARGET_MET short-circuit
    skips the Aβ append for the final observation)."""
    lg = _make_terminated_loop()
    p = build_payload(lg)
    assert "per_iteration" in p
    pit = p["per_iteration"]
    assert pit["truncated"] is False
    assert pit["cap"] == 256
    iters = p["loop"]["iterations_used"]
    assert len(pit["error_history"]) == iters
    # Aβ has at most iterations_used - 1 entries; loops that terminate
    # on TARGET_MET have one fewer (the short-circuit skips the append).
    assert len(pit["convergence_profile"]) <= iters - 1
    assert len(pit["convergence_profile"]) >= iters - 2


def test_payload_per_iteration_truncates_at_cap():
    """Loops longer than PER_ITERATION_CAP are truncated; truncated flag set."""
    from loopgain.telemetry import PER_ITERATION_CAP

    # Drive a long-running CONVERGING loop: Aβ ≈ 0.7 throughout, which
    # stays under the STALLING threshold so the loop never terminates on
    # OSCILLATING. target_error=None disables the short-circuit so the
    # geometric decay never triggers TARGET_MET; max_iterations caps it
    # past PER_ITERATION_CAP so the trajectory exceeds the cap.
    n = PER_ITERATION_CAP + 50
    lg = LoopGain(target_error=None, max_iterations=n)
    err = 1.0
    for _ in range(n):
        if not lg.should_continue():
            break
        lg.observe(err)
        err *= 0.7
    p = build_payload(lg)
    pit = p["per_iteration"]
    assert pit["truncated"] is True
    assert len(pit["error_history"]) == PER_ITERATION_CAP
    assert len(pit["convergence_profile"]) == PER_ITERATION_CAP


def test_payload_per_iteration_excludes_outputs():
    """Per-iteration arrays must not contain customer outputs even when
    they were passed to observe()."""
    big_output = {"prompt": "secret customer data"}
    lg = LoopGain(target_error=0.5, max_iterations=10)
    for e in [10.0, 5.0, 0.3]:
        if not lg.should_continue():
            break
        lg.observe(e, output=big_output)
    p = build_payload(lg)
    p_json = json.dumps(p)
    assert "secret customer data" not in p_json
    # error_history entries are floats, not output objects.
    for entry in p["per_iteration"]["error_history"]:
        assert isinstance(entry, (int, float))


def test_payload_includes_classification_fields_when_provided():
    lg = _make_terminated_loop()
    p = build_payload(
        lg,
        framework="langgraph",
        loop_type="verify_revise",
        team="search-prod",
    )
    assert p["framework"] == "langgraph"
    assert p["loop_type"] == "verify_revise"
    assert p["team"] == "search-prod"


def test_payload_classification_fields_default_to_none():
    lg = _make_terminated_loop()
    p = build_payload(lg)
    assert p["framework"] is None
    assert p["loop_type"] is None
    assert p["team"] is None


def test_send_telemetry_passes_classification_fields(monkeypatch):
    """LoopGain.send_telemetry plumbs framework/loop_type/team through."""

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    ok = lg.send_telemetry(
        endpoint="https://telemetry.loopgain.ai/v1/aggregate",
        token="t",
        framework="crewai",
        loop_type="rag_refine",
        team="ml-team",
    )
    assert ok is True
    body = json.loads(captured["body"])
    assert body["framework"] == "crewai"
    assert body["loop_type"] == "rag_refine"
    assert body["team"] == "ml-team"


def test_send_telemetry_can_disable_per_iteration(monkeypatch):
    """include_per_iteration=False is plumbed through send_telemetry."""

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    ok = lg.send_telemetry(
        endpoint="https://telemetry.loopgain.ai/v1/aggregate",
        token="t",
        include_per_iteration=False,
    )
    assert ok is True
    body = json.loads(captured["body"])
    assert "per_iteration" not in body


# ----- send_telemetry pass-through tests (existing) -----


def test_send_telemetry_method_passes_through_allow_insecure(monkeypatch):
    """LoopGain.send_telemetry plumbs allow_insecure through to send_payload."""

    class FakeResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return FakeResponse()

    monkeypatch.setattr("loopgain.telemetry._open_request", fake_urlopen)

    lg = _make_terminated_loop()
    # Without allow_insecure, http:// is rejected and urlopen is never called.
    ok = lg.send_telemetry("http://localhost:8787/v1/aggregate", token="t")
    assert ok is False
    assert "url" not in captured

    # With allow_insecure=True, the request goes through.
    ok = lg.send_telemetry(
        "http://localhost:8787/v1/aggregate",
        token="t",
        allow_insecure=True,
    )
    assert ok is True
    assert captured["url"] == "http://localhost:8787/v1/aggregate"


def test_send_payload_refuses_redirects():
    """The bearer token must never be sent across a 30x redirect.

    Regression: ``urllib`` follows redirects by default and does NOT strip
    the Authorization header on cross-origin hops. If the configured
    endpoint were compromised, a 302 to ``attacker.com`` would harvest
    the token. ``_open_request`` uses a no-redirect opener so any 3xx
    surfaces as a failed delivery instead of a leak.
    """
    import io
    import urllib.error
    import urllib.request

    from loopgain.telemetry import _NoRedirectHandler

    handler = _NoRedirectHandler()
    # Each of the standard redirect codes must raise an HTTPError, which
    # `send_payload`'s outer `except Exception:` then converts to `False`.
    for method in (
        handler.http_error_301,
        handler.http_error_302,
        handler.http_error_303,
        handler.http_error_307,
        handler.http_error_308,
    ):
        req = urllib.request.Request("https://example.com/")
        with pytest.raises(urllib.error.HTTPError):
            method(req, io.BytesIO(b""), 302, "Found", {})


# ----- send_payload retry behavior (transient failures) -----

import socket as _socket
import urllib.error as _uerr

from loopgain import telemetry as _tele


class _OkResp:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _retry_payload():
    return build_payload(_make_terminated_loop(), workload_id="retry-test")


def test_send_payload_retries_transient_then_succeeds(monkeypatch):
    """A transient failure (timeout) is retried; a later success returns True."""
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _socket.timeout("slow first attempts")
        return _OkResp()

    sleeps: list[float] = []
    monkeypatch.setattr("loopgain.telemetry._open_request", flaky)
    monkeypatch.setattr("loopgain.telemetry.time.sleep", lambda s: sleeps.append(s))

    ok = send_payload("https://t.example/v1/aggregate", token="t", payload=_retry_payload())
    assert ok is True
    assert calls["n"] == 3                 # two transient failures, third succeeds
    assert sleeps == [0.25, 0.5]           # linear backoff between attempts


def test_send_payload_gives_up_after_retries_on_persistent_5xx(monkeypatch):
    """A persistent transient (503) exhausts retries and returns False."""
    calls = {"n": 0}

    def always_503(req, timeout=None):
        calls["n"] += 1
        raise _uerr.HTTPError("https://t.example", 503, "unavailable", {}, None)

    monkeypatch.setattr("loopgain.telemetry._open_request", always_503)
    monkeypatch.setattr("loopgain.telemetry.time.sleep", lambda s: None)

    ok = send_payload("https://t.example/v1/aggregate", token="t", payload=_retry_payload(), retries=2)
    assert ok is False
    assert calls["n"] == 3                  # 1 initial + 2 retries


def test_send_payload_does_not_retry_deterministic_4xx(monkeypatch):
    """A 401 will never succeed on retry — fail fast, no backoff."""
    calls = {"n": 0}
    slept = {"n": 0}

    def unauthorized(req, timeout=None):
        calls["n"] += 1
        raise _uerr.HTTPError("https://t.example", 401, "unauthorized", {}, None)

    monkeypatch.setattr("loopgain.telemetry._open_request", unauthorized)
    monkeypatch.setattr("loopgain.telemetry.time.sleep", lambda s: slept.__setitem__("n", slept["n"] + 1))

    ok = send_payload("https://t.example/v1/aggregate", token="bad", payload=_retry_payload())
    assert ok is False
    assert calls["n"] == 1                  # no retry on a deterministic 4xx
    assert slept["n"] == 0


def test_send_payload_retries_zero_is_single_shot(monkeypatch):
    """retries=0 restores the original single-attempt behavior."""
    calls = {"n": 0}

    def timeout(req, timeout=None):
        calls["n"] += 1
        raise TimeoutError()

    monkeypatch.setattr("loopgain.telemetry._open_request", timeout)
    monkeypatch.setattr("loopgain.telemetry.time.sleep", lambda s: None)

    ok = send_payload("https://t.example/v1/aggregate", token="t", payload=_retry_payload(), retries=0)
    assert ok is False
    assert calls["n"] == 1


def test_send_payload_never_raises_on_unexpected_error(monkeypatch):
    """A non-transient, unexpected error is swallowed (best-effort), no retry."""
    def boom(req, timeout=None):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("loopgain.telemetry._open_request", boom)
    monkeypatch.setattr("loopgain.telemetry.time.sleep", lambda s: None)

    assert send_payload("https://t.example/v1/aggregate", token="t", payload=_retry_payload()) is False


def test_is_transient_classification():
    assert _tele._is_transient(TimeoutError()) is True
    assert _tele._is_transient(_socket.timeout()) is True
    assert _tele._is_transient(_uerr.URLError("dns")) is True
    assert _tele._is_transient(_uerr.HTTPError("u", 503, "x", {}, None)) is True
    assert _tele._is_transient(_uerr.HTTPError("u", 429, "x", {}, None)) is True
    assert _tele._is_transient(_uerr.HTTPError("u", 400, "x", {}, None)) is False
    assert _tele._is_transient(_uerr.HTTPError("u", 401, "x", {}, None)) is False
    assert _tele._is_transient(RuntimeError("x")) is False
