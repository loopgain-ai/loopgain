"""Tests for the framework integration adapters.

The adapters delegate iteration to LangGraph / CrewAI / AutoGen but the
contract under test is purely the LoopGain → framework glue. We mock the
framework surface (``stream``, ``astream``, ``step_callback``,
``run_stream``) so this test module needs no framework installs.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from loopgain import LoopGain
from loopgain.integrations import AutoGenAdapter, CrewAIAdapter, LangGraphAdapter


# ── Lazy-import surface ───────────────────────────────────────────────


def test_integrations_package_does_not_eagerly_import_frameworks():
    """Importing loopgain.integrations must not pull in langgraph, crewai,
    or autogen — they're optional deps and importing the package is cheap."""
    import sys

    for forbidden in ("langgraph", "crewai", "autogen", "autogen_agentchat"):
        assert forbidden not in sys.modules, (
            f"loopgain.integrations imported {forbidden!r}; "
            "adapters must lazy-import their frameworks"
        )


def test_unknown_attribute_raises_attribute_error():
    import loopgain.integrations as integ

    with pytest.raises(AttributeError, match="no attribute 'NopeAdapter'"):
        integ.NopeAdapter  # type: ignore[attr-defined]


def test_framework_name_constants():
    assert LangGraphAdapter.framework_name == "langgraph"
    assert CrewAIAdapter.framework_name == "crewai"
    assert AutoGenAdapter.framework_name == "autogen"


# ── LangGraph adapter ─────────────────────────────────────────────────


class _FakeLangGraph:
    """Minimal stand-in for a compiled LangGraph. Yields the configured
    items from ``stream`` / ``astream`` and records the kwargs."""

    def __init__(self, updates: List[Any]):
        self.updates = updates
        self.last_kwargs: dict[str, Any] = {}

    def stream(self, input, config=None, stream_mode=None, **kwargs):
        self.last_kwargs = {"input": input, "config": config, "stream_mode": stream_mode, **kwargs}
        for u in self.updates:
            yield u

    async def astream(self, input, config=None, stream_mode=None, **kwargs):
        self.last_kwargs = {"input": input, "config": config, "stream_mode": stream_mode, **kwargs}
        for u in self.updates:
            yield u


def test_langgraph_adapter_observes_each_step():
    updates = [
        {"verifier": {"errors": [1, 2, 3]}},
        {"verifier": {"errors": [1]}},
        {"verifier": {"errors": []}},  # zero errors → target met
    ]
    graph = _FakeLangGraph(updates)
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = LangGraphAdapter(
        lg=lg,
        error_fn=lambda u: len(u["verifier"]["errors"]),
    )
    final = adapter.run(graph, {"draft": "x"}, config={"thread_id": "t1"})
    assert final == updates[-1]
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"
    # config + stream_mode were passed through.
    assert graph.last_kwargs["config"] == {"thread_id": "t1"}
    assert graph.last_kwargs["stream_mode"] == "updates"


def test_langgraph_adapter_stops_when_loopgain_terminates():
    """Once LoopGain hits a terminal state (TARGET_MET here), the
    adapter must stop pulling from the graph stream even if more events
    are queued."""
    updates = [{"e": 5}, {"e": 1}, {"e": 0.1}, {"e": "would-blow-up-if-pulled"}]
    graph = _FakeLangGraph(updates)
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = LangGraphAdapter(lg=lg, error_fn=lambda u: float(u["e"]) if isinstance(u["e"], (int, float)) else 0.0)
    list(adapter.stream(graph, {}))
    # Adapter consumed exactly 3 items (third one hits target_error).
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"


def test_langgraph_adapter_skips_step_when_error_fn_returns_none():
    """error_fn returning None must NOT call observe — useful for
    skipping setup/init steps that don't carry an error signal yet."""
    updates = [{"setup": True}, {"e": 1.0}, {"e": 0.5}]
    graph = _FakeLangGraph(updates)
    lg = LoopGain(target_error=0.0, max_iterations=20)
    adapter = LangGraphAdapter(
        lg=lg,
        error_fn=lambda u: None if "setup" in u else float(u["e"]),
    )
    list(adapter.stream(graph, {}))
    # Only 2 observations even though 3 items streamed.
    assert lg.result.iterations_used == 2


def test_langgraph_adapter_async_path_uses_astream():
    updates = [{"e": 4}, {"e": 2}, {"e": 0.1}]
    graph = _FakeLangGraph(updates)
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = LangGraphAdapter(lg=lg, error_fn=lambda u: float(u["e"]))
    final = asyncio.run(adapter.arun(graph, {}))
    assert final == updates[-1]
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"


def test_langgraph_adapter_async_error_fn_overrides_sync():
    updates = [{"e": 4}, {"e": 0.1}]
    graph = _FakeLangGraph(updates)
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = LangGraphAdapter(lg=lg, error_fn=lambda u: 999.0)  # would not converge

    async def aerror(u):
        return float(u["e"])

    asyncio.run(adapter.arun(graph, {}, error_fn=aerror))
    assert lg.result.outcome == "converged"


# ── CrewAI adapter ────────────────────────────────────────────────────


class _FakeCrew:
    """Stand-in for crewai.Crew with the two callback attributes."""

    def __init__(self):
        self.step_callback = None
        self.task_callback = None


def test_crewai_adapter_requires_at_least_one_error_fn():
    lg = LoopGain()
    with pytest.raises(ValueError, match="at least one observation source"):
        CrewAIAdapter(lg=lg)


def test_crewai_adapter_install_wires_callbacks():
    lg = LoopGain(target_error=0.5, max_iterations=20)
    crew = _FakeCrew()
    adapter = CrewAIAdapter(lg=lg, step_error_fn=lambda step: float(step["e"]))
    adapter.install(crew)

    assert crew.step_callback is not None
    # Drive the callback as CrewAI would.
    crew.step_callback({"e": 3.0})
    crew.step_callback({"e": 1.0})
    crew.step_callback({"e": 0.1})  # below target → terminate
    crew.step_callback({"e": "ignored"})  # post-terminal: must be a no-op
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"


def test_crewai_adapter_uninstall_restores_originals():
    lg = LoopGain()
    crew = _FakeCrew()
    sentinel_calls: list[Any] = []

    def original(step: Any) -> None:
        sentinel_calls.append(step)

    crew.step_callback = original
    adapter = CrewAIAdapter(lg=lg, step_error_fn=lambda s: 1.0)
    adapter.install(crew)
    # Install must wrap, not replace — calling the wrapped callback should
    # also invoke the original so existing instrumentation isn't lost.
    crew.step_callback({"e": 1.0})
    assert sentinel_calls == [{"e": 1.0}]
    adapter.uninstall()
    assert crew.step_callback is original


def test_crewai_adapter_context_manager_uninstalls():
    lg = LoopGain()
    crew = _FakeCrew()
    crew.step_callback = None
    with CrewAIAdapter(lg=lg, step_error_fn=lambda s: 1.0) as adapter:
        adapter.install(crew)
        assert crew.step_callback is not None
    # On context exit, original (None) is restored.
    assert crew.step_callback is None


def test_crewai_adapter_task_callback_path():
    lg = LoopGain(target_error=0.5, max_iterations=20)
    crew = _FakeCrew()
    adapter = CrewAIAdapter(lg=lg, task_error_fn=lambda out: float(out["score"]))
    adapter.install(crew)
    crew.task_callback({"score": 5.0})
    crew.task_callback({"score": 0.1})
    assert lg.result.iterations_used == 2
    assert lg.result.outcome == "converged"


def test_crewai_adapter_chained_callback_swallows_user_exceptions():
    """If the user's existing callback raises, the adapter's observation
    must still happen — keeping LoopGain's view of the loop intact."""
    lg = LoopGain()
    crew = _FakeCrew()

    def buggy(step: Any) -> None:
        raise RuntimeError("user callback exploded")

    crew.step_callback = buggy
    adapter = CrewAIAdapter(lg=lg, step_error_fn=lambda s: 1.0)
    adapter.install(crew)
    crew.step_callback({"e": 1.0})  # should not raise
    assert lg.result.iterations_used == 1


# ── AutoGen adapter ───────────────────────────────────────────────────


class _FakeMessage:
    def __init__(self, source: str, content: Any):
        self.source = source
        self.content = content


class _FakeTaskResult:
    """Mimics autogen's TaskResult duck shape (messages + stop_reason)."""

    def __init__(self, messages, stop_reason):
        self.messages = messages
        self.stop_reason = stop_reason


class _FakeTeam:
    def __init__(self, messages: List[Any]):
        self.messages = messages
        self.last_task: Any = None

    def run_stream(self, *, task=None, cancellation_token=None):
        self.last_task = task
        # Need to return an async iterator.
        async def gen():
            for m in self.messages:
                yield m

        return gen()


class _FakeCancellationToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_autogen_adapter_observes_filtered_messages():
    msgs = [
        _FakeMessage("generator", "draft v1"),
        _FakeMessage("verifier", 5.0),
        _FakeMessage("generator", "draft v2"),
        _FakeMessage("verifier", 1.0),
        _FakeMessage("generator", "draft v3"),
        _FakeMessage("verifier", 0.1),  # converges
        _FakeTaskResult(messages=[], stop_reason="done"),
    ]
    team = _FakeTeam(msgs)
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = AutoGenAdapter(
        lg=lg,
        error_fn=lambda m: float(m.content),
        observe_sources={"verifier"},
    )
    out = asyncio.run(adapter.run(team, task="hi"))
    assert out[0] is msgs[0]
    # Only verifier messages drive observe(); 3 of them.
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"


def test_autogen_adapter_skips_task_result_for_observation():
    """The terminal TaskResult must be yielded but NOT sent to error_fn."""
    msgs = [
        _FakeMessage("verifier", 0.1),
        _FakeTaskResult(messages=[], stop_reason="done"),
    ]
    team = _FakeTeam(msgs)
    lg = LoopGain(target_error=0.5, max_iterations=20)

    seen: list[Any] = []

    def err(m):
        seen.append(m)
        return float(m.content) if hasattr(m, "content") and isinstance(m.content, (int, float)) else 0.0

    adapter = AutoGenAdapter(lg=lg, error_fn=err)
    asyncio.run(adapter.run(team, task="hi"))
    # The TaskResult must not have reached error_fn even though we
    # didn't filter by source.
    for s in seen:
        assert not isinstance(s, _FakeTaskResult)


def test_autogen_adapter_cancels_token_on_terminal_state():
    msgs = [
        _FakeMessage("verifier", 0.1),  # immediate convergence
        _FakeMessage("verifier", "would-blow-up-if-observed"),
    ]
    team = _FakeTeam(msgs)
    token = _FakeCancellationToken()
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = AutoGenAdapter(
        lg=lg,
        error_fn=lambda m: float(m.content) if isinstance(m.content, (int, float)) else 0.0,
        observe_sources={"verifier"},
    )
    asyncio.run(adapter.run(team, task="hi", cancellation_token=token))
    assert token.cancelled is True


def test_autogen_adapter_async_error_fn():
    msgs = [_FakeMessage("verifier", 0.1)]
    team = _FakeTeam(msgs)
    lg = LoopGain(target_error=0.5, max_iterations=20)

    async def aerror(m):
        return float(m.content)

    adapter = AutoGenAdapter(lg=lg, error_fn=aerror, observe_sources={"verifier"})  # type: ignore[arg-type]
    asyncio.run(adapter.run(team, task="hi"))
    assert lg.result.outcome == "converged"
