"""LangChain adapter for LoopGain.

Wraps any LangChain agent that exposes ``stream(input, ...)`` /
``astream(input, ...)`` with a LoopGain monitor. Each yielded chunk is
treated as one iteration: the user's ``error_fn`` is called with the
chunk, the returned magnitude is fed to ``LoopGain.observe()``, and
iteration terminates whenever LoopGain detects a stop condition (target
met, oscillation, divergence, max iterations).

Supports both LangChain's current ``langchain.agents.create_agent()``
(the v1+ pre-built agent, which is a LangGraph graph under the hood)
and the legacy ``AgentExecutor``. The adapter is duck-typed: any object
with ``stream``/``astream`` returning an iterator works. The shape of
each yielded chunk depends on which API the user constructed and what
``stream_mode`` / ``version`` kwargs they pass:

- ``create_agent`` with ``stream_mode="updates", version="v2"`` yields
  ``{"type": "updates", "data": {step_name: state_update}}`` per step.
- ``create_agent`` with ``stream_mode="updates"`` (no ``version``)
  yields the inner update dict directly per step.
- Legacy ``AgentExecutor.stream(...)`` yields an ``AddableDict`` per
  step (intermediate ``actions`` / ``steps`` keys, then a final dict
  with the ``output`` key).

The adapter forwards ``**stream_kwargs`` to the framework's stream
method without interpretation, so the user controls the chunk shape
their ``error_fn`` will receive.

References:
- https://docs.langchain.com/oss/python/langchain/agents
- https://docs.langchain.com/oss/python/langchain/streaming
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Iterator, Optional

if TYPE_CHECKING:
    from loopgain.core import LoopGain


ErrorFn = Callable[[Any], Optional[float]]
AsyncErrorFn = Callable[[Any], Awaitable[Optional[float]]]


class LangChainAdapter:
    """Drive a LangChain agent with a LoopGain monitor.

    Args:
        lg: A ``LoopGain`` instance to drive. The adapter calls
            ``lg.observe()`` on each non-None error returned by
            ``error_fn`` and stops iterating once
            ``lg.should_continue()`` returns False.
        error_fn: Maps one stream chunk to an error magnitude (or
            ``None`` to skip). The chunk's shape depends on what
            ``stream_mode`` / ``version`` you pass to ``stream``; for
            ``create_agent`` with ``stream_mode="updates", version="v2"``
            the chunk is ``{"type": "updates", "data": {...}}``. For
            legacy ``AgentExecutor`` it's an ``AddableDict``. The user
            owns the parse.

    Example (modern ``create_agent``)::

        from langchain.agents import create_agent
        from loopgain import LoopGain
        from loopgain.integrations import LangChainAdapter

        agent = create_agent(model="gpt-5-nano", tools=[get_weather])
        lg = LoopGain(target_error=0.0, max_iterations=20)

        def error_fn(chunk):
            # stream_mode="updates", version="v2": one chunk per step.
            if chunk.get("type") != "updates":
                return None
            # Count unresolved tool calls — drops to 0 once the agent
            # finishes calling tools and returns a final answer.
            return sum(
                1 for _, update in chunk["data"].items()
                if getattr(update.get("messages", [None])[-1], "tool_calls", None)
            )

        adapter = LangChainAdapter(lg=lg, error_fn=error_fn)
        final = adapter.run(
            agent,
            {"messages": [{"role": "user", "content": "What's the weather?"}]},
            stream_mode="updates",
            version="v2",
        )

    Example (legacy ``AgentExecutor``)::

        from langchain.agents import AgentExecutor
        executor = AgentExecutor(agent=..., tools=[...])
        lg = LoopGain(target_error=0.1, max_iterations=20)

        adapter = LangChainAdapter(
            lg=lg,
            # Each chunk is an AddableDict with `intermediate_steps`,
            # `actions`, or a terminal `output` key.
            error_fn=lambda chunk: len(chunk.get("intermediate_steps", [])),
        )
        final = adapter.run(executor, {"input": "Find the bug."})
    """

    framework_name = "langchain"

    def __init__(
        self,
        lg: "LoopGain",
        error_fn: ErrorFn,
    ) -> None:
        self.lg = lg
        self.error_fn = error_fn

    def run(
        self,
        agent: Any,
        input: Any,
        **stream_kwargs: Any,
    ) -> Any:
        """Drive ``agent.stream()`` synchronously, returning the last
        yielded chunk. Callers wanting the full trace should iterate
        ``stream()`` directly."""
        last: Any = None
        for item in self.stream(agent, input, **stream_kwargs):
            last = item
        return last

    def stream(
        self,
        agent: Any,
        input: Any,
        **stream_kwargs: Any,
    ) -> Iterator[Any]:
        """Yield each stream chunk while driving LoopGain.

        Iteration stops as soon as LoopGain reaches a terminal state,
        even if the underlying agent would have produced more events.
        ``**stream_kwargs`` is forwarded verbatim — pass ``stream_mode``,
        ``version``, ``config`` etc. as your agent expects them.
        """
        iterator = agent.stream(input, **stream_kwargs)
        for item in iterator:
            if not self.lg.should_continue():
                break
            magnitude = self.error_fn(item)
            if magnitude is not None:
                self.lg.observe(magnitude, output=item)
            yield item

    async def arun(
        self,
        agent: Any,
        input: Any,
        error_fn: Optional[AsyncErrorFn] = None,
        **stream_kwargs: Any,
    ) -> Any:
        """Async counterpart of ``run``. Drives ``agent.astream()``.

        If ``error_fn`` is provided here it overrides the sync one; this
        is the entry point for callers whose error derivation is async
        (e.g. an LLM-as-judge call).
        """
        last: Any = None
        async for item in self.astream(agent, input, error_fn=error_fn, **stream_kwargs):
            last = item
        return last

    async def astream(
        self,
        agent: Any,
        input: Any,
        error_fn: Optional[AsyncErrorFn] = None,
        **stream_kwargs: Any,
    ) -> AsyncIterator[Any]:
        iterator = agent.astream(input, **stream_kwargs)
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
