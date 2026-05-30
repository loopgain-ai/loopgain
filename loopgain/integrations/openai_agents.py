"""OpenAI Agents SDK adapter for LoopGain.

Wraps the OpenAI Agents SDK's ``Runner.run_streamed(agent, input)`` with
a LoopGain monitor. The SDK is async-first: streaming is only available
through the async ``RunResultStreaming.stream_events()`` API, so this
adapter's primary surface is async (``run`` / ``stream``). A sync
``run_sync`` helper wraps the async path with ``asyncio.run`` for
callers in synchronous code paths.

``stream_events()`` yields three kinds of events:

- ``raw_response_event`` — token-level LLM deltas. Rarely an iteration
  boundary; usually excluded from observation via ``observe_event_types``.
- ``run_item_stream_event`` — coarse "an item was produced" event
  (``tool_call_item``, ``tool_call_output_item``, ``message_output_item``).
  This is the natural iteration unit for most loops.
- ``agent_updated_stream_event`` — fires when control hands off to a
  different agent. Typically not observed but kept in the yielded
  stream for the caller's awareness.

By default the adapter only forwards ``run_item_stream_event`` to
``error_fn`` (the most common case for verify-revise / tool-use loops);
override with ``observe_event_types=None`` to observe every event or
pass a different ``set[str]``.

After the stream completes, ``result.final_output`` carries the agent's
final answer. ``run()`` returns the ``RunResultStreaming`` so callers
can access ``final_output`` and any other post-run state.

Reference: https://openai.github.io/openai-agents-python/streaming/
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from loopgain.core import LoopGain


EventErrorFn = Callable[[Any], Optional[float]]
AsyncEventErrorFn = Callable[[Any], Awaitable[Optional[float]]]

# Default observation set: high-level item events are the iteration unit
# for most agent loops. Raw token deltas and agent-handoff notifications
# are passed through to the yielded stream but not fed to ``error_fn``.
DEFAULT_OBSERVE_EVENT_TYPES = frozenset({"run_item_stream_event"})


class OpenAIAgentsAdapter:
    """Drive an OpenAI Agents SDK ``Agent`` with a LoopGain monitor.

    Args:
        lg: A ``LoopGain`` instance to drive.
        error_fn: Maps one stream event to an error magnitude. Both
            sync and async callables are accepted. Return ``None`` to
            skip an event. Only events whose ``.type`` is in
            ``observe_event_types`` are forwarded to ``error_fn``;
            others are yielded but not observed.
        observe_event_types: Set of ``StreamEvent.type`` strings to
            forward to ``error_fn``. Defaults to
            ``{"run_item_stream_event"}``. Pass ``None`` to observe
            every event regardless of type.

    Example::

        from agents import Agent, Runner, function_tool, ItemHelpers
        from loopgain import LoopGain
        from loopgain.integrations import OpenAIAgentsAdapter

        agent = Agent(name="Reviser", instructions="...", tools=[...])

        lg = LoopGain(target_error=0.0, max_iterations=20)

        def error_fn(event):
            # run_item_stream_event with type=tool_call_output_item:
            # parse the verifier's reported failure count.
            if event.item.type == "tool_call_output_item":
                output = event.item.output
                # e.g. verifier returns {"failures": N}
                return float(output.get("failures", 0))
            return None

        adapter = OpenAIAgentsAdapter(lg=lg, error_fn=error_fn)
        result = await adapter.run(agent, input="Fix the bug.")
        print(result.final_output)
    """

    framework_name = "openai-agents"

    def __init__(
        self,
        lg: "LoopGain",
        error_fn: EventErrorFn,
        observe_event_types: Optional[set[str]] = None,
    ) -> None:
        self.lg = lg
        self.error_fn = error_fn

        from loopgain import funnel

        funnel.note_adapter(self.framework_name)
        # ``None`` means "observe every event". ``frozenset()`` ≠ ``None``.
        if observe_event_types is None:
            self.observe_event_types: Optional[frozenset[str]] = None
        else:
            self.observe_event_types = (
                DEFAULT_OBSERVE_EVENT_TYPES
                if observe_event_types is DEFAULT_OBSERVE_EVENT_TYPES
                else frozenset(observe_event_types)
            )

    async def run(
        self,
        agent: Any,
        input: Any,
        **run_kwargs: Any,
    ) -> Any:
        """Drive ``Runner.run_streamed(agent, input)`` to completion,
        returning the ``RunResultStreaming`` (which carries
        ``final_output``, ``new_items``, etc.).

        If LoopGain reaches a terminal state mid-stream the adapter
        breaks out of ``stream_events()`` and best-effort cancels the
        underlying run (the SDK's ``RunResultStreaming.cancel()`` is
        called if available).
        """
        from agents import Runner

        result = Runner.run_streamed(agent, input, **run_kwargs)
        async for _ in self._drive(result):
            pass
        return result

    async def stream(
        self,
        agent: Any,
        input: Any,
        **run_kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Yield each ``StreamEvent`` from ``Runner.run_streamed`` while
        driving LoopGain. The terminal ``RunResultStreaming`` itself is
        not yielded — access it via ``adapter.run(...)`` which returns
        it after consuming the stream."""
        from agents import Runner

        result = Runner.run_streamed(agent, input, **run_kwargs)
        async for event in self._drive(result):
            yield event

    def run_sync(
        self,
        agent: Any,
        input: Any,
        **run_kwargs: Any,
    ) -> Any:
        """Synchronous wrapper around ``run``. Calls ``asyncio.run`` —
        do not call from inside a running event loop. Streaming is
        still active under the hood; only the caller's interface is
        synchronous.
        """
        return asyncio.run(self.run(agent, input, **run_kwargs))

    async def _drive(self, result: Any) -> AsyncIterator[Any]:
        """Iterate ``result.stream_events()`` and feed observations to
        LoopGain. Cancels the run when LoopGain reaches a terminal
        state."""
        cancelled = False
        try:
            async for event in result.stream_events():
                yield event

                event_type = getattr(event, "type", None)
                if (
                    self.observe_event_types is not None
                    and event_type not in self.observe_event_types
                ):
                    continue

                magnitude = self.error_fn(event)
                if hasattr(magnitude, "__await__"):
                    magnitude = await magnitude  # type: ignore[assignment]

                if magnitude is not None:
                    self.lg.observe(magnitude, output=event)

                if not self.lg.should_continue():
                    cancel = getattr(result, "cancel", None)
                    if callable(cancel):
                        try:
                            cancel()
                            cancelled = True
                        except Exception:
                            # Best-effort: if cancel raises (e.g. the SDK
                            # changed signatures), just stop pulling.
                            pass
                    break
        except asyncio.CancelledError:
            if not cancelled:
                raise
