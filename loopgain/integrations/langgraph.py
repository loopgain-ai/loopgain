"""LangGraph adapter for LoopGain.

Wraps a compiled LangGraph (anything exposing ``stream(input, ...)`` /
``astream(input, ...)``) with a LoopGain monitor. Each step yielded by
``graph.stream(input, stream_mode="updates")`` is treated as one iteration:
the user's ``error_fn`` is called with the per-step update dict, the
returned magnitude is fed to ``LoopGain.observe()``, and the iteration
terminates whenever LoopGain detects a stop condition (target met,
oscillation, divergence, max iterations).

Reference: https://reference.langchain.com/python/langgraph/pregel/main/Pregel/stream

LangGraph's ``stream_mode`` options (see context7 docs for full set):

- ``"values"``   — full state after each step
- ``"updates"``  — node names and their per-step updates  (default for adapter)
- ``"tasks"``    — task start/finish events with results/errors
- ``"messages"`` — token-by-token LLM stream
- ``"checkpoints"``, ``"custom"``, ``"debug"``

The adapter defaults to ``"updates"`` because (a) one yielded item per
graph step lines up with LoopGain's iteration model and (b) the per-node
update dict is the smallest object that carries enough state for an
``error_fn`` to make a decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Iterator, Optional

if TYPE_CHECKING:
    from loopgain.core import LoopGain


# Single error-derivation signature shared by all adapters. Returning None
# means "no error signal this step" — the iteration is consumed but
# LoopGain.observe() is not called (useful for ignoring setup steps).
ErrorFn = Callable[[Any], Optional[float]]
AsyncErrorFn = Callable[[Any], Awaitable[Optional[float]]]


class LangGraphAdapter:
    """Drive a LangGraph compiled graph with a LoopGain monitor.

    Args:
        lg: A ``LoopGain`` instance to drive. The adapter calls
            ``lg.observe()`` on each non-None error returned by ``error_fn``
            and stops iterating once ``lg.should_continue()`` returns False.
        error_fn: Maps one stream item to an error magnitude (or ``None`` to
            skip). For ``stream_mode="updates"`` (the default), the item is a
            dict mapping node name → state-update dict. Return any
            non-negative number; ``len(errors)`` is a common pattern.
        stream_mode: Pass-through to ``graph.stream(stream_mode=...)``.
            Defaults to ``"updates"``. See LangGraph docs for other modes.

    Example::

        from langgraph.graph import StateGraph
        from loopgain import LoopGain
        from loopgain.integrations import LangGraphAdapter

        graph = build_my_verify_revise_graph().compile()
        lg = LoopGain(target_error=0.1, max_iterations=20)
        adapter = LangGraphAdapter(
            lg=lg,
            error_fn=lambda update: len(update.get("verifier", {}).get("errors", [])),
        )
        final_state = adapter.run(graph, {"draft": initial})
    """

    framework_name = "langgraph"

    def __init__(
        self,
        lg: "LoopGain",
        error_fn: ErrorFn,
        stream_mode: str = "updates",
    ) -> None:
        self.lg = lg
        self.error_fn = error_fn
        self.stream_mode = stream_mode

    def run(
        self,
        graph: Any,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        **stream_kwargs: Any,
    ) -> Any:
        """Drive ``graph.stream()`` synchronously, returning the last item.

        ``config`` is forwarded to ``graph.stream(config=...)``; any
        additional kwargs are forwarded too. The final yielded item is
        returned; callers wanting the full trace should iterate
        ``stream()`` directly.
        """
        last: Any = None
        for item in self.stream(graph, input, config=config, **stream_kwargs):
            last = item
        return last

    def stream(
        self,
        graph: Any,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        **stream_kwargs: Any,
    ) -> Iterator[Any]:
        """Yield each stream item while driving LoopGain.

        Iteration stops as soon as LoopGain reaches a terminal state, even
        if the underlying graph would have produced more events. The user
        is responsible for any post-stop cleanup.
        """
        iterator = graph.stream(
            input, config=config, stream_mode=self.stream_mode, **stream_kwargs
        )
        for item in iterator:
            if not self.lg.should_continue():
                break
            magnitude = self.error_fn(item)
            if magnitude is not None:
                self.lg.observe(magnitude, output=item)
            yield item

    async def arun(
        self,
        graph: Any,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        error_fn: Optional[AsyncErrorFn] = None,
        **stream_kwargs: Any,
    ) -> Any:
        """Async counterpart of ``run``. Drives ``graph.astream()``.

        If ``error_fn`` is provided here it overrides the sync one; this is
        the entry point for callers whose error derivation is async (e.g.
        an LLM-as-judge call).
        """
        last: Any = None
        async for item in self.astream(graph, input, config=config, error_fn=error_fn, **stream_kwargs):
            last = item
        return last

    async def astream(
        self,
        graph: Any,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        error_fn: Optional[AsyncErrorFn] = None,
        **stream_kwargs: Any,
    ) -> AsyncIterator[Any]:
        iterator = graph.astream(
            input, config=config, stream_mode=self.stream_mode, **stream_kwargs
        )
        async for item in iterator:
            if not self.lg.should_continue():
                break
            magnitude: Optional[float]
            if error_fn is not None:
                magnitude = await error_fn(item)
            else:
                magnitude = self.error_fn(item)
            if magnitude is not None:
                self.lg.observe(magnitude, output=item)
            yield item
