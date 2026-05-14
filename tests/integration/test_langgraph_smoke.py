"""Real-framework smoke for the LangGraph adapter.

Skipped automatically if `langgraph` isn't installed. Run via:

    pip install 'loopgain[langgraph]'
    pytest tests/integration -m integration

The test builds a tiny ``StateGraph`` that decrements an ``error`` field
in its single node and loops back conditionally until the error reaches
the target. The adapter drives the graph with a ``LoopGain`` whose
``target_error`` should fire convergence in exactly N iterations.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

langgraph = pytest.importorskip("langgraph")

from typing import TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402

from loopgain import LoopGain  # noqa: E402
from loopgain.integrations import LangGraphAdapter  # noqa: E402


class State(TypedDict):
    error: float


def _build_graph():
    """Geometric-decay loop: error_{n+1} = error_n * 0.5. Loops back to
    `revise` until error <= 0.05; then routes to END."""

    def revise(state: State) -> State:
        return {"error": state["error"] * 0.5}

    def should_continue(state: State) -> str:
        return END if state["error"] <= 0.05 else "revise"

    graph = StateGraph(State)
    graph.add_node("revise", revise)
    graph.add_edge(START, "revise")
    graph.add_conditional_edges("revise", should_continue, {END: END, "revise": "revise"})
    return graph.compile()


def test_langgraph_adapter_drives_real_graph_to_convergence():
    """The adapter must observe each loop iteration of a real LangGraph
    StateGraph, drive LoopGain to convergence, and return a sensible
    final state. The loop converges in 5 iterations: 1 → 0.5 → 0.25 →
    0.125 → 0.0625 → 0.03125 (target_error=0.05 met)."""
    graph = _build_graph()
    lg = LoopGain(target_error=0.05, max_iterations=15)

    adapter = LangGraphAdapter(
        lg=lg,
        # stream_mode="updates" yields {node_name: state_update_dict}.
        # Pull the new error from the revise node's update.
        error_fn=lambda update: update["revise"]["error"],
    )

    final = adapter.run(graph, {"error": 1.0}, config={"recursion_limit": 50})

    assert final is not None
    # 5 observe() calls: 0.5, 0.25, 0.125, 0.0625, 0.03125 (target).
    assert lg.result.iterations_used == 5
    assert lg.result.outcome == "converged"
    assert lg.result.best_error == pytest.approx(0.03125)
    # Aβ should be ~0.5 throughout (geometric decay with ratio 0.5).
    assert all(0.4 < ab < 0.6 for ab in lg.result.convergence_profile)


def test_langgraph_adapter_stream_mode_passes_through_to_graph():
    """The adapter accepts arbitrary stream_mode values. Verify a
    different mode (`values`) still produces observable items."""
    graph = _build_graph()
    lg = LoopGain(target_error=0.05, max_iterations=15)

    # stream_mode="values" yields the full state dict per step.
    adapter = LangGraphAdapter(
        lg=lg,
        error_fn=lambda state: state.get("error") if isinstance(state, dict) else None,
        stream_mode="values",
    )
    items = list(adapter.stream(graph, {"error": 1.0}, config={"recursion_limit": 50}))
    assert len(items) > 0
    assert lg.result.outcome == "converged"


def test_langgraph_adapter_telemetry_payload_stamps_framework():
    """End-to-end: drive a real graph, build a telemetry payload from the
    resulting LoopGain, confirm framework stamp lands. No network call."""
    from loopgain.telemetry import build_payload

    graph = _build_graph()
    lg = LoopGain(target_error=0.05, max_iterations=15)
    adapter = LangGraphAdapter(
        lg=lg,
        error_fn=lambda update: update["revise"]["error"],
    )
    adapter.run(graph, {"error": 1.0}, config={"recursion_limit": 50})

    payload = build_payload(lg, workload_id="ig-smoke", framework=adapter.framework_name)
    assert payload["framework"] == "langgraph"
    assert payload["workload_id"] == "ig-smoke"
    assert payload["loop"]["outcome"] == "converged"
    assert payload["loop"]["iterations_used"] == 5
    # Per-iteration trajectories made it through (schema v3 default).
    assert "per_iteration" in payload
    assert len(payload["per_iteration"]["error_history"]) == 5
