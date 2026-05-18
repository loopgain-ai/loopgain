"""Real-framework smoke for the LangChain adapter.

Skipped automatically if `langchain` isn't installed. Run via:

    pip install 'loopgain[langchain]'
    pytest tests/integration -m integration

The LangChain adapter is duck-typed against anything exposing
``stream(input, **kwargs)`` / ``astream(input, **kwargs)``. We don't
want this test to require an LLM API key, so we drive a tiny
hand-rolled object that mimics LangChain's stream surface (an
``AddableDict``-ish stream of step chunks). The real ``langchain``
import gate above still ensures we only run this when the framework
is installed — catching ImportError drift or any unconditional
top-level imports the adapter accidentally introduces.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator

import pytest

pytestmark = pytest.mark.integration

langchain = pytest.importorskip("langchain")

from loopgain import LoopGain  # noqa: E402
from loopgain.integrations import LangChainAdapter  # noqa: E402


class _FakeStreamingAgent:
    """Minimal duck-type of a LangChain agent's stream surface.

    Each call to ``stream(input, **kwargs)`` yields one chunk per
    "step", where each chunk is a dict shaped like LangChain's update
    stream: ``{"type": "updates", "data": {"step_name": {...}}}``.
    The ``error`` field decreases geometrically — five chunks bring it
    from 1.0 to ~0.03125, below ``target_error=0.05``.
    """

    SEQUENCE = [0.5, 0.25, 0.125, 0.0625, 0.03125]

    def stream(self, input: Any, **kwargs: Any) -> Iterator[Any]:
        for err in self.SEQUENCE:
            yield {"type": "updates", "data": {"reviser": {"error": err}}}

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        for err in self.SEQUENCE:
            yield {"type": "updates", "data": {"reviser": {"error": err}}}


def _error_from_chunk(chunk):
    """error_fn for the adapter: pull the float out of update chunks."""
    if chunk.get("type") != "updates":
        return None
    for _step, update in chunk["data"].items():
        if "error" in update:
            return float(update["error"])
    return None


def test_langchain_adapter_drives_fake_agent_to_convergence():
    """The adapter must observe each yielded chunk, drive LoopGain to
    convergence at the third step (0.125 still > 0.05; 0.0625 still >
    0.05; 0.03125 ≤ 0.05 → converged in 5 iterations)."""
    agent = _FakeStreamingAgent()
    lg = LoopGain(target_error=0.05, max_iterations=15)
    adapter = LangChainAdapter(lg=lg, error_fn=_error_from_chunk)

    final = adapter.run(agent, {"messages": [{"role": "user", "content": "go"}]})

    assert final is not None
    assert lg.result.iterations_used == 5
    assert lg.result.outcome == "converged"
    assert lg.result.best_error == pytest.approx(0.03125)


def test_langchain_adapter_stream_yields_all_chunks_when_not_terminal():
    """Streaming with a higher target should yield every chunk and
    terminate cleanly at iterator exhaustion."""
    agent = _FakeStreamingAgent()
    lg = LoopGain(target_error=0.0, max_iterations=20)  # target won't fire
    adapter = LangChainAdapter(lg=lg, error_fn=_error_from_chunk)

    items = list(adapter.stream(agent, {"messages": []}))
    assert len(items) == 5
    # No terminal condition met — outcome is whichever stability state
    # the run ended in (could be CONVERGING for a clean geometric decay).
    assert lg.result.iterations_used == 5


def test_langchain_adapter_async_path():
    """The async path must mirror the sync path: same observations,
    same termination after 5 iterations."""

    async def main():
        agent = _FakeStreamingAgent()
        lg = LoopGain(target_error=0.05, max_iterations=15)
        adapter = LangChainAdapter(lg=lg, error_fn=_error_from_chunk)
        final = await adapter.arun(agent, {"messages": []})
        return lg, final

    lg, final = asyncio.run(main())
    assert lg.result.outcome == "converged"
    assert lg.result.iterations_used == 5
    assert final is not None


def test_langchain_adapter_telemetry_payload_stamps_framework():
    """End-to-end: drive the fake agent, build a telemetry payload, confirm
    framework stamp lands. No network call."""
    from loopgain.telemetry import build_payload

    agent = _FakeStreamingAgent()
    lg = LoopGain(target_error=0.05, max_iterations=15)
    adapter = LangChainAdapter(lg=lg, error_fn=_error_from_chunk)
    adapter.run(agent, {"messages": []})

    payload = build_payload(lg, workload_id="lc-smoke", framework=adapter.framework_name)
    assert payload["framework"] == "langchain"
    assert payload["workload_id"] == "lc-smoke"
    assert payload["loop"]["outcome"] == "converged"
    assert payload["loop"]["iterations_used"] == 5
    assert "per_iteration" in payload
    assert len(payload["per_iteration"]["error_history"]) == 5
