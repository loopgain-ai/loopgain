"""Real-framework smoke for the OpenAI Agents SDK adapter.

Skipped automatically if `agents` (the import surface of the
``openai-agents`` pip package) isn't installed. Run via:

    pip install 'loopgain[openai-agents]'
    pytest tests/integration -m integration

The Agents SDK is fundamentally async-streaming and requires either a
real OpenAI key or an internal ``Runner`` fake to exercise the streaming
path. We don't want this test to require network or API budget, so we
construct a tiny ``RunResultStreaming`` lookalike that emits a sequence
of ``StreamEvent``-shaped objects via an async generator, then drive
the adapter against it. The framework import gate above still ensures
the test runs only when the SDK is installed (so any unconditional
top-level import drift in the adapter would surface here).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

pytestmark = pytest.mark.integration

agents = pytest.importorskip("agents")

from loopgain import LoopGain  # noqa: E402
from loopgain.integrations import OpenAIAgentsAdapter  # noqa: E402


class _FakeStreamingResult:
    """Duck-type of ``agents.result.RunResultStreaming``.

    Exposes ``stream_events()`` (async generator), ``cancel()`` (which
    sets a flag the test can assert on), and ``final_output`` (set
    after the stream completes).
    """

    # (event_type, magnitude) — verifier-style declining scores.
    EVENTS = [
        ("raw_response_event", None),       # raw deltas, ignored by default
        ("run_item_stream_event", 5.0),     # tool output 1
        ("agent_updated_stream_event", None),
        ("run_item_stream_event", 1.0),     # tool output 2
        ("run_item_stream_event", 0.3),     # tool output 3 → target_error=0.5 met
    ]

    def __init__(self):
        self.cancelled = False
        self.final_output = "done"

    async def stream_events(self) -> AsyncIterator[Any]:
        for event_type, magnitude in self.EVENTS:
            if self.cancelled:
                return
            event = SimpleNamespace(type=event_type, magnitude=magnitude)
            yield event

    def cancel(self):
        self.cancelled = True


def _error_from_event(event) -> float | None:
    """Adapter's error_fn: pull the magnitude attached to events we
    care about. Returns None to skip events without a magnitude."""
    return event.magnitude


def test_openai_agents_adapter_drives_fake_result_to_convergence(monkeypatch):
    """Patch ``Runner.run_streamed`` to return our fake, run the
    adapter, confirm it observes only ``run_item_stream_event`` types
    (the default) and converges after the third score (0.3 ≤ 0.5)."""

    fake = _FakeStreamingResult()

    def fake_run_streamed(agent, input, **kwargs):
        return fake

    from agents import Runner
    monkeypatch.setattr(Runner, "run_streamed", staticmethod(fake_run_streamed))

    async def main():
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = OpenAIAgentsAdapter(lg=lg, error_fn=_error_from_event)
        result = await adapter.run(agent=object(), input="ignored")
        return lg, result

    lg, result = asyncio.run(main())

    # 3 run_item observations: 5.0, 1.0, 0.3 (target met).
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"
    # Adapter should have called .cancel() on terminal state.
    assert fake.cancelled is True
    # The result returned by .run() is the underlying RunResultStreaming.
    assert result is fake


def test_openai_agents_adapter_observe_event_types_none_observes_all(monkeypatch):
    """With observe_event_types=None, every event reaches error_fn —
    including raw_response_event and agent_updated_stream_event. The
    raw_response_event has magnitude=None so it's skipped, but the
    agent_updated_stream_event would also be passed to the fn."""

    seen_types: list[str] = []

    def tracking_fn(event):
        seen_types.append(event.type)
        return event.magnitude

    fake = _FakeStreamingResult()

    def fake_run_streamed(agent, input, **kwargs):
        return fake

    from agents import Runner
    monkeypatch.setattr(Runner, "run_streamed", staticmethod(fake_run_streamed))

    async def main():
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = OpenAIAgentsAdapter(
            lg=lg, error_fn=tracking_fn, observe_event_types=None
        )
        await adapter.run(agent=object(), input="ignored")
        return lg

    lg = asyncio.run(main())

    # All five event types reached the fn (including raw + agent_updated).
    assert "raw_response_event" in seen_types
    assert "agent_updated_stream_event" in seen_types
    # Still converges on the run_item events with non-None magnitudes.
    assert lg.result.outcome == "converged"


def test_openai_agents_adapter_telemetry_payload_stamps_framework(monkeypatch):
    """End-to-end: drive the fake, build a telemetry payload, confirm
    framework stamp lands."""
    fake = _FakeStreamingResult()

    def fake_run_streamed(agent, input, **kwargs):
        return fake

    from agents import Runner
    monkeypatch.setattr(Runner, "run_streamed", staticmethod(fake_run_streamed))

    async def main():
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = OpenAIAgentsAdapter(lg=lg, error_fn=_error_from_event)
        await adapter.run(agent=object(), input="ignored")
        return lg, adapter

    lg, adapter = asyncio.run(main())

    from loopgain.telemetry import build_payload

    payload = build_payload(
        lg, workload_id="oa-smoke", framework=adapter.framework_name
    )
    assert payload["framework"] == "openai-agents"
    assert payload["loop"]["outcome"] == "converged"
    assert payload["loop"]["iterations_used"] == 3
