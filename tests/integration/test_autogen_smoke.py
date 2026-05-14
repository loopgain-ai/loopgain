"""Real-framework smoke for the AutoGen v0.4+ adapter.

Skipped automatically if `autogen_agentchat` isn't installed. Run via:

    pip install 'loopgain[autogen]'
    pytest tests/integration -m integration

We construct a real ``RoundRobinGroupChat`` with two custom
``BaseChatAgent`` subclasses (no real LLM, no network). The verifier
agent emits a strictly-decreasing numeric "score"; the adapter watches
for it and drives LoopGain to convergence. This exercises the full
async ``team.run_stream`` path against the real framework runtime,
catching any drift in message types, source filtering, or the terminal
TaskResult shape.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

pytestmark = pytest.mark.integration

autogen_agentchat = pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import BaseChatAgent  # noqa: E402
from autogen_agentchat.base import Response  # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination  # noqa: E402
from autogen_agentchat.messages import BaseChatMessage, TextMessage  # noqa: E402
from autogen_agentchat.teams import RoundRobinGroupChat  # noqa: E402
from autogen_core import CancellationToken  # noqa: E402

from loopgain import LoopGain  # noqa: E402
from loopgain.integrations import AutoGenAdapter  # noqa: E402


class _Generator(BaseChatAgent):
    """Trivial generator: emits a TextMessage saying which iteration we
    are on. Carries no error signal — the adapter must skip it."""

    def __init__(self, name: str = "generator"):
        super().__init__(name=name, description="emits drafts")
        self._n = 0

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        self._n += 1
        return Response(chat_message=TextMessage(content=f"draft v{self._n}", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._n = 0


class _Verifier(BaseChatAgent):
    """Trivial verifier: emits a TextMessage whose content is a
    monotonically-decreasing float. The adapter's error_fn parses it
    and feeds it to LoopGain."""

    SEQUENCE = [5.0, 1.0, 0.3, 0.05]

    def __init__(self, name: str = "verifier"):
        super().__init__(name=name, description="scores drafts")
        self._i = 0

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        score = self.SEQUENCE[min(self._i, len(self.SEQUENCE) - 1)]
        self._i += 1
        return Response(chat_message=TextMessage(content=str(score), source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._i = 0


def _parse_score(message) -> float | None:
    """error_fn for the adapter: pull the float out of verifier messages."""
    try:
        return float(message.content)
    except (ValueError, AttributeError, TypeError):
        return None


def test_autogen_adapter_drives_real_team_to_convergence():
    """Drive a real RoundRobinGroupChat. The verifier's third score
    (0.3) is below target_error=0.5, which should fire LoopGain
    convergence after 3 verifier observations."""

    async def main():
        team = RoundRobinGroupChat(
            participants=[_Generator(), _Verifier()],
            termination_condition=MaxMessageTermination(max_messages=20),
        )
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = AutoGenAdapter(
            lg=lg,
            error_fn=_parse_score,
            observe_sources={"verifier"},
        )
        token = CancellationToken()
        out = await adapter.run(team, task="Start.", cancellation_token=token)
        return lg, out, token

    lg, out, token = asyncio.run(main())

    assert lg.result.outcome == "converged"
    # 3 verifier observations: 5.0, 1.0, 0.3 (target met).
    assert lg.result.iterations_used == 3
    # Adapter should have cancelled the run at terminal state. AutoGen's
    # CancellationToken exposes a method, not an attribute — we documented
    # the duck-typed `cancel()` call earlier so this is the integration
    # check that the real type matches what the adapter assumes.
    assert token.is_cancelled() is True
    # We broke out of the iteration before the framework emitted its
    # final TaskResult wrapper, so out's last element is the verifier's
    # converging message rather than a TaskResult. Either is acceptable
    # — the contract is just "we yielded what the framework emitted up
    # to the cancellation point."
    assert len(out) >= 1


def test_autogen_adapter_does_not_observe_generator_messages():
    """With observe_sources={'verifier'}, generator messages must be
    yielded but never reach error_fn. Verified by tracking what the
    error_fn sees."""

    seen: list = []

    def tracking_fn(msg):
        seen.append(getattr(msg, "source", None))
        return _parse_score(msg)

    async def main():
        team = RoundRobinGroupChat(
            participants=[_Generator(), _Verifier()],
            termination_condition=MaxMessageTermination(max_messages=20),
        )
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = AutoGenAdapter(
            lg=lg,
            error_fn=tracking_fn,
            observe_sources={"verifier"},
        )
        await adapter.run(team, task="Start.", cancellation_token=CancellationToken())
        return lg

    lg = asyncio.run(main())
    # Only verifier messages reached the error_fn.
    assert all(src == "verifier" for src in seen), f"saw non-verifier sources: {seen}"
    assert lg.result.outcome == "converged"


def test_autogen_adapter_telemetry_payload_stamps_framework():
    async def main():
        team = RoundRobinGroupChat(
            participants=[_Generator(), _Verifier()],
            termination_condition=MaxMessageTermination(max_messages=20),
        )
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = AutoGenAdapter(
            lg=lg,
            error_fn=_parse_score,
            observe_sources={"verifier"},
        )
        await adapter.run(team, task="Start.", cancellation_token=CancellationToken())
        return lg, adapter

    lg, adapter = asyncio.run(main())

    from loopgain.telemetry import build_payload

    payload = build_payload(lg, workload_id="autogen-smoke", framework=adapter.framework_name)
    assert payload["framework"] == "autogen"
    assert payload["loop"]["outcome"] == "converged"
