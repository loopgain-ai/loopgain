"""Real-framework smoke for the CrewAI adapter.

Skipped automatically if `crewai` isn't installed, or if the local Python
is incompatible with crewai's pin (it currently requires <3.14). Run via:

    pip install 'loopgain[crewai]'
    pytest tests/integration -m integration

We construct a Crew and a Task, install the adapter, and invoke the
``step_callback`` / ``task_callback`` directly the way CrewAI does
internally. This avoids spinning up a real Agent loop (which would
require an LLM key and network access) while still exercising the
adapter against the **actual** Crew object's attribute surface — which
is what catches the kind of API-drift mocks miss.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

crewai = pytest.importorskip("crewai")

from crewai import Agent, Crew, Task  # noqa: E402

from loopgain import LoopGain  # noqa: E402
from loopgain.integrations import CrewAIAdapter  # noqa: E402


def _build_crew():
    """Build a real Crew with a single dummy Agent and Task. The Crew is
    not kicked off — we drive its callbacks directly to keep the test
    LLM-free and offline."""
    agent = Agent(
        role="Echo agent",
        goal="Repeat the input verbatim.",
        backstory="A trivial test agent that does nothing useful.",
        # llm parameter omitted → CrewAI accepts but kickoff would fail;
        # we never call kickoff() in this test.
        allow_delegation=False,
    )
    task = Task(
        description="Echo this string back.",
        expected_output="The input string verbatim.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    return crew


def test_crewai_adapter_installs_on_real_crew():
    """The adapter installs callbacks on a real crewai.Crew instance.
    The Crew object's ``step_callback`` / ``task_callback`` attributes
    must be writable post-construction."""
    crew = _build_crew()
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = CrewAIAdapter(
        lg=lg,
        step_error_fn=lambda step: float(step["e"]),
        task_error_fn=lambda out: float(out["score"]),
    )
    adapter.install(crew)
    # Both callbacks should now be set to the adapter's chained handler.
    assert crew.step_callback is not None
    assert crew.task_callback is not None
    adapter.uninstall()


def test_crewai_adapter_drives_loopgain_via_callbacks():
    """Invoke the installed callbacks the way CrewAI's executor would,
    confirm LoopGain reaches a terminal state. This is the closest we
    can get to an end-to-end test without a real LLM."""
    crew = _build_crew()
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = CrewAIAdapter(
        lg=lg,
        step_error_fn=lambda step: float(step["e"]),
    )
    adapter.install(crew)

    # Simulate three agent steps with decreasing error.
    crew.step_callback({"e": 4.0})
    crew.step_callback({"e": 1.0})
    crew.step_callback({"e": 0.3})  # below target_error

    assert lg.result.iterations_used == 3
    assert lg.result.outcome == "converged"
    adapter.uninstall()


def test_crewai_adapter_chains_with_user_callback_on_real_crew():
    """User-provided ``step_callback`` set in the Crew constructor must
    survive: the adapter wraps it, doesn't replace it. We verify by
    constructing a Crew with a sentinel callback already in place."""
    captured: list[dict] = []

    def sentinel(step):
        captured.append(step)

    agent = Agent(
        role="Echo agent",
        goal="Echo input.",
        backstory="Test agent.",
        allow_delegation=False,
    )
    task = Task(
        description="Echo.",
        expected_output="Echoed.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], step_callback=sentinel)
    assert crew.step_callback is sentinel  # baseline

    lg = LoopGain()
    adapter = CrewAIAdapter(lg=lg, step_error_fn=lambda step: float(step["e"]))
    adapter.install(crew)
    crew.step_callback({"e": 1.0})
    # User's sentinel ran *and* LoopGain observed.
    assert captured == [{"e": 1.0}]
    assert lg.result.iterations_used == 1
    adapter.uninstall()
    # Original is restored.
    assert crew.step_callback is sentinel


def test_crewai_adapter_telemetry_payload_stamps_framework():
    crew = _build_crew()
    lg = LoopGain(target_error=0.5, max_iterations=20)
    adapter = CrewAIAdapter(
        lg=lg,
        step_error_fn=lambda step: float(step["e"]),
    )
    adapter.install(crew)
    crew.step_callback({"e": 5.0})
    crew.step_callback({"e": 1.0})
    crew.step_callback({"e": 0.1})

    from loopgain.telemetry import build_payload

    payload = build_payload(lg, workload_id="crewai-smoke", framework=adapter.framework_name)
    assert payload["framework"] == "crewai"
    assert payload["loop"]["outcome"] == "converged"
    assert payload["loop"]["iterations_used"] == 3
    adapter.uninstall()
