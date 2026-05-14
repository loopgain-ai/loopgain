"""AutoGen v0.4+ adapter for LoopGain.

AutoGen v0.4 reorganized around an event-driven async runtime: a Team
(``RoundRobinGroupChat``, ``SocietyOfMindAgent``, ``Swarm``, etc.) exposes
``run_stream(task=...)`` which yields ``BaseAgentEvent | BaseChatMessage``
items per message, terminating with a ``TaskResult``.

In a verify-revise pattern the Team is typically a 2-agent rotation
(generator → verifier → generator → ...). The verifier's most recent
message carries the error signal; the user's ``error_fn`` extracts it
and the adapter feeds it to LoopGain.

Reference: https://microsoft.github.io/autogen/stable/_modules/autogen_agentchat/teams/_group_chat/_base_group_chat.html

The adapter does NOT support the legacy v0.2 ``ConversableAgent.initiate_chat``
API. v0.2 is in maintenance mode upstream; users on the old runtime
should upgrade or fall back to the raw ``LoopGain.observe()`` loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, List, Optional

if TYPE_CHECKING:
    from loopgain.core import LoopGain


AsyncMessageErrorFn = Callable[[Any], Awaitable[Optional[float]]]
MessageErrorFn = Callable[[Any], Optional[float]]


class AutoGenAdapter:
    """Drive an AutoGen v0.4+ Team with a LoopGain monitor.

    Args:
        lg: A ``LoopGain`` instance to drive.
        error_fn: Maps one streamed message/event to an error magnitude.
            Return ``None`` to skip (e.g. for non-verifier messages).
            Both sync and async callables are accepted; if async, await
            in the function body — the adapter will detect and handle.
        observe_sources: Optional set of agent ``source`` names to observe.
            If provided, messages from other sources are passed through
            without invoking ``error_fn``. Useful when only the verifier
            agent's messages carry an error signal.

    Example::

        from autogen_agentchat.teams import RoundRobinGroupChat
        from loopgain import LoopGain
        from loopgain.integrations import AutoGenAdapter

        team = RoundRobinGroupChat(participants=[generator, verifier])
        lg = LoopGain(target_error=0.1, max_iterations=20)
        adapter = AutoGenAdapter(
            lg=lg,
            error_fn=lambda msg: parse_verifier_score(msg.content),
            observe_sources={"verifier"},
        )
        result = await adapter.run(team, task="Write a haiku about loops.")
    """

    framework_name = "autogen"

    def __init__(
        self,
        lg: "LoopGain",
        error_fn: MessageErrorFn,
        observe_sources: Optional[set[str]] = None,
    ) -> None:
        self.lg = lg
        self.error_fn = error_fn
        self.observe_sources = observe_sources

    async def run(
        self,
        team: Any,
        task: Any,
        cancellation_token: Optional[Any] = None,
    ) -> List[Any]:
        """Drive ``team.run_stream(task=...)`` to completion, returning
        the full list of yielded messages/events (including the terminal
        ``TaskResult``).

        If LoopGain reaches a terminal state mid-stream, the team is
        cancelled via the supplied ``cancellation_token`` (if one was
        provided) — AutoGen has no way to interrupt a stream from outside
        the cancellation-token mechanism.
        """
        out: List[Any] = []
        async for item in self.stream(team, task, cancellation_token=cancellation_token):
            out.append(item)
        return out

    async def stream(
        self,
        team: Any,
        task: Any,
        cancellation_token: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """Yield each message/event from ``team.run_stream`` while driving
        LoopGain. Cancels the team's cancellation_token when LoopGain
        signals a terminal state."""
        kwargs: dict[str, Any] = {"task": task}
        if cancellation_token is not None:
            kwargs["cancellation_token"] = cancellation_token

        async for message in team.run_stream(**kwargs):
            yield message

            # Don't observe the terminal TaskResult — it's a wrapper, not
            # a per-iteration event. Detect by duck-typing on the
            # `messages` + `stop_reason` attributes (AutoGen's TaskResult
            # shape) so we don't have to import the framework.
            if hasattr(message, "messages") and hasattr(message, "stop_reason"):
                continue

            # Source-filter: skip messages we're not configured to observe.
            if self.observe_sources is not None:
                source = getattr(message, "source", None)
                if source not in self.observe_sources:
                    continue

            magnitude = self.error_fn(message)
            # Allow the user to write either a sync or async error_fn.
            if hasattr(magnitude, "__await__"):
                magnitude = await magnitude  # type: ignore[assignment]

            if magnitude is not None:
                self.lg.observe(magnitude, output=message)

            if not self.lg.should_continue() and cancellation_token is not None:
                # Best-effort: AutoGen uses the cancellation token to abort.
                cancel = getattr(cancellation_token, "cancel", None)
                if callable(cancel):
                    cancel()
