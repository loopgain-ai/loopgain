"""Claude Agent SDK adapter for LoopGain.

Wraps Anthropic's ``claude_agent_sdk`` with a LoopGain monitor. The SDK
is async-only: ``query(prompt=..., options=...)`` returns an async
iterator of messages, each of which is one of:

- ``UserMessage`` — the user-supplied prompt echoed back
- ``AssistantMessage`` — model output with content blocks (``TextBlock``,
  ``ToolUseBlock``)
- ``SystemMessage`` — system events
- ``ResultMessage`` — terminal message with summary fields (cost, usage)

The natural iteration unit for an agent loop is one ``AssistantMessage``
(or one full tool-call → tool-result round-trip). The user's
``error_fn`` decides which messages carry an error signal — typically
by inspecting ``AssistantMessage.content`` for self-reported state or
counting unresolved ``ToolUseBlock`` entries.

The adapter accepts either:

- a ``prompt`` (string) and optional ``options`` — the adapter
  constructs the ``query(...)`` iterator itself; or
- a pre-constructed ``message_iterator`` (e.g. from
  ``ClaudeSDKClient.receive_messages()`` or ``receive_response()``) —
  the adapter just drives it.

By default the adapter only forwards ``AssistantMessage`` instances to
``error_fn`` (since user/system messages don't typically carry an error
signal). Override with ``observe_message_types=None`` to observe every
message, or pass a tuple of types to widen the filter.

Reference: https://github.com/anthropics/claude-agent-sdk-python
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Optional, Tuple

if TYPE_CHECKING:
    from loopgain.core import LoopGain


MessageErrorFn = Callable[[Any], Optional[float]]
AsyncMessageErrorFn = Callable[[Any], Awaitable[Optional[float]]]


def _default_observe_types() -> Tuple[type, ...]:
    """Default observation filter: ``AssistantMessage`` only.

    Imported lazily so the adapter module itself stays importable
    without ``claude_agent_sdk`` installed (importing the package is
    what raises ``ImportError`` from ``run()`` if it's missing).
    """
    from claude_agent_sdk import AssistantMessage

    return (AssistantMessage,)


class ClaudeAgentSDKAdapter:
    """Drive a Claude Agent SDK ``query`` or ``ClaudeSDKClient`` message
    stream with a LoopGain monitor.

    Args:
        lg: A ``LoopGain`` instance to drive.
        error_fn: Maps one yielded message to an error magnitude. Both
            sync and async callables are accepted. Return ``None`` to
            skip a message. Only messages of a type in
            ``observe_message_types`` are forwarded to ``error_fn``;
            others are yielded but not observed.
        observe_message_types: Tuple of message classes to forward to
            ``error_fn``. Defaults to ``(AssistantMessage,)``. Pass
            ``None`` to observe every message regardless of type. The
            tuple is resolved lazily on first stream so the module
            stays importable without ``claude_agent_sdk`` installed.

    Example::

        from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, TextBlock
        from loopgain import LoopGain
        from loopgain.integrations import ClaudeAgentSDKAdapter

        def error_fn(message):
            # Count `FAIL:` markers the verifier-persona emits.
            for block in getattr(message, "content", []):
                if isinstance(block, TextBlock):
                    return float(block.text.count("FAIL:"))
            return None

        lg = LoopGain(target_error=0.0, max_iterations=20)
        adapter = ClaudeAgentSDKAdapter(lg=lg, error_fn=error_fn)

        options = ClaudeAgentOptions(system_prompt="Self-verify each draft.")
        result = await adapter.run(
            prompt="Write a haiku about feedback loops.",
            options=options,
        )
    """

    framework_name = "claude-agent-sdk"

    def __init__(
        self,
        lg: "LoopGain",
        error_fn: MessageErrorFn,
        observe_message_types: Optional[Tuple[type, ...]] = (),
    ) -> None:
        self.lg = lg
        self.error_fn = error_fn
        # Sentinel: empty tuple → use defaults on first stream.
        # ``None`` → observe everything. Otherwise the user-supplied
        # tuple is honored verbatim.
        self._observe_types_arg = observe_message_types
        self._resolved_observe_types: Optional[Tuple[type, ...]] = None

    def _resolve_observe_types(self) -> Optional[Tuple[type, ...]]:
        if self._observe_types_arg is None:
            return None
        if self._resolved_observe_types is None:
            if self._observe_types_arg == ():
                self._resolved_observe_types = _default_observe_types()
            else:
                self._resolved_observe_types = tuple(self._observe_types_arg)
        return self._resolved_observe_types

    async def run(
        self,
        prompt: Optional[Any] = None,
        *,
        options: Optional[Any] = None,
        message_iterator: Optional[AsyncIterator[Any]] = None,
    ) -> list:
        """Drive a message stream to completion, returning the full
        list of yielded messages.

        Exactly one of ``prompt`` (with optional ``options``) or
        ``message_iterator`` must be supplied:

        - With ``prompt``, the adapter constructs the iterator via
          ``claude_agent_sdk.query(prompt=..., options=...)``.
        - With ``message_iterator``, the adapter drives whatever async
          iterator is passed — e.g. ``ClaudeSDKClient.receive_messages()``.
        """
        out: list = []
        async for message in self.stream(
            prompt=prompt, options=options, message_iterator=message_iterator
        ):
            out.append(message)
        return out

    async def stream(
        self,
        prompt: Optional[Any] = None,
        *,
        options: Optional[Any] = None,
        message_iterator: Optional[AsyncIterator[Any]] = None,
    ) -> AsyncIterator[Any]:
        """Yield each message from the underlying stream while driving
        LoopGain. Stops iterating as soon as LoopGain reaches a
        terminal state.
        """
        if (prompt is None) == (message_iterator is None):
            raise ValueError(
                "ClaudeAgentSDKAdapter.stream/run requires exactly one of "
                "`prompt` or `message_iterator`."
            )

        if message_iterator is None:
            from claude_agent_sdk import query

            message_iterator = query(prompt=prompt, options=options)

        observe_types = self._resolve_observe_types()

        async for message in message_iterator:
            yield message

            if observe_types is not None and not isinstance(message, observe_types):
                continue

            magnitude = self.error_fn(message)
            if hasattr(magnitude, "__await__"):
                magnitude = await magnitude  # type: ignore[assignment]

            if magnitude is not None:
                self.lg.observe(magnitude, output=message)

            if not self.lg.should_continue():
                # The SDK has no caller-facing cancel for a query()
                # iterator; breaking out drops our subscription and
                # the underlying transport tears down on garbage
                # collection. ClaudeSDKClient users should call
                # ``client.disconnect()`` after the stream returns.
                break

    def run_sync(
        self,
        prompt: Optional[Any] = None,
        *,
        options: Optional[Any] = None,
    ) -> list:
        """Synchronous wrapper around ``run`` for the ``prompt`` form.
        Calls ``asyncio.run`` — do not call from inside a running event
        loop. The bidirectional ``message_iterator`` form is async-only.
        """
        return asyncio.run(self.run(prompt=prompt, options=options))
