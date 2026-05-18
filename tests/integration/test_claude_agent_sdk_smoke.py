"""Real-framework smoke for the Claude Agent SDK adapter.

Skipped automatically if `claude_agent_sdk` isn't installed. Run via:

    pip install 'loopgain[claude-agent-sdk]'
    pytest tests/integration -m integration

The SDK's ``query(prompt=..., options=...)`` reaches out to a local
Claude transport that wants an API key + network — not what we want
for a CI-grade smoke. We pass our own async iterator of ``AssistantMessage``
instances (constructed from the real SDK types) to the adapter via the
``message_iterator`` parameter, exercising the adapter's drive logic
end-to-end against the framework's actual message types. The
``claude_agent_sdk`` import gate still ensures we only run when the
SDK is installed.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

pytestmark = pytest.mark.integration

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, SystemMessage, TextBlock  # noqa: E402

from loopgain import LoopGain  # noqa: E402
from loopgain.integrations import ClaudeAgentSDKAdapter  # noqa: E402


async def _make_message_stream(scores: list[float | None]) -> AsyncIterator[Any]:
    """Yield AssistantMessages whose first TextBlock carries the score
    (or a SystemMessage for None entries, which should be skipped by
    the default observe-types filter)."""
    for score in scores:
        if score is None:
            try:
                yield SystemMessage(subtype="info", data={"note": "system event"})
            except TypeError:
                # Older SDK SystemMessage signatures — fall back to a
                # minimally-constructed instance via __new__ so we still
                # exercise the type filter without depending on the
                # exact constructor surface.
                msg = SystemMessage.__new__(SystemMessage)
                yield msg
        else:
            try:
                yield AssistantMessage(
                    content=[TextBlock(text=f"score: {score}")],
                    model="claude-test",
                )
            except TypeError:
                # Constructor drift: build via __new__ + setattr.
                msg = AssistantMessage.__new__(AssistantMessage)
                msg.content = [TextBlock(text=f"score: {score}")]
                yield msg


def _error_from_message(message) -> float | None:
    """Parse `score: X` from the first TextBlock."""
    for block in getattr(message, "content", []) or []:
        if isinstance(block, TextBlock):
            text = block.text
            if "score:" in text:
                try:
                    return float(text.split("score:")[1].strip())
                except (ValueError, IndexError):
                    return None
    return None


def test_claude_agent_sdk_adapter_drives_message_iterator_to_convergence():
    """Drive a real-type message stream — SystemMessages mixed in are
    skipped by the default AssistantMessage-only filter; the three
    AssistantMessage scores drive LoopGain to convergence."""

    async def main():
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = ClaudeAgentSDKAdapter(lg=lg, error_fn=_error_from_message)
        stream = _make_message_stream([5.0, None, 1.0, None, 0.3])
        result = await adapter.run(message_iterator=stream)
        return lg, result

    lg, result = asyncio.run(main())

    # 3 AssistantMessage observations: 5.0, 1.0, 0.3 (target met).
    # SystemMessages were yielded but not observed.
    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"
    # All five messages yielded through.
    assert len(result) == 5


def test_claude_agent_sdk_adapter_observe_types_none_observes_all():
    """With observe_message_types=None, every message reaches error_fn —
    including SystemMessage. SystemMessages have no `score:` text so
    they return None, but the fn must still be called on them."""

    seen: list[type] = []

    def tracking_fn(message):
        seen.append(type(message))
        return _error_from_message(message)

    async def main():
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = ClaudeAgentSDKAdapter(
            lg=lg, error_fn=tracking_fn, observe_message_types=None
        )
        stream = _make_message_stream([5.0, None, 1.0, None, 0.3])
        await adapter.run(message_iterator=stream)
        return lg

    lg = asyncio.run(main())

    # SystemMessage was seen by the fn (would not have been with default filter).
    assert any(t is SystemMessage for t in seen)
    assert lg.result.outcome == "converged"


def test_claude_agent_sdk_adapter_requires_exactly_one_input():
    """``run`` raises ValueError when both / neither prompt and
    message_iterator are supplied."""

    async def main_both():
        adapter = ClaudeAgentSDKAdapter(lg=LoopGain(), error_fn=lambda m: None)
        with pytest.raises(ValueError):
            await adapter.run(prompt="hello", message_iterator=_make_message_stream([]))

    async def main_neither():
        adapter = ClaudeAgentSDKAdapter(lg=LoopGain(), error_fn=lambda m: None)
        with pytest.raises(ValueError):
            await adapter.run()

    asyncio.run(main_both())
    asyncio.run(main_neither())


def test_claude_agent_sdk_adapter_telemetry_payload_stamps_framework():
    """End-to-end: drive the stream, build a telemetry payload, confirm
    framework stamp lands."""
    async def main():
        lg = LoopGain(target_error=0.5, max_iterations=20)
        adapter = ClaudeAgentSDKAdapter(lg=lg, error_fn=_error_from_message)
        stream = _make_message_stream([5.0, 1.0, 0.3])
        await adapter.run(message_iterator=stream)
        return lg, adapter

    lg, adapter = asyncio.run(main())

    from loopgain.telemetry import build_payload

    payload = build_payload(
        lg, workload_id="cas-smoke", framework=adapter.framework_name
    )
    assert payload["framework"] == "claude-agent-sdk"
    assert payload["loop"]["outcome"] == "converged"
    assert payload["loop"]["iterations_used"] == 3
