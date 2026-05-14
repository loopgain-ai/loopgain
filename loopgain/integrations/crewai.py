"""CrewAI adapter for LoopGain.

CrewAI's iteration model differs from LangGraph: a Crew is a sequence of
Tasks, each executed by an Agent that may iterate internally (tool calls,
self-correction, retries). Two callback hooks are exposed by the public API:

- ``step_callback`` (Agent or Crew level) — fires once per agent thought/
  step. Receives an ``AgentAction`` / ``AgentFinish`` / ``ToolResult``-shaped
  object containing the prompt, thought, tool, tool_input, and result.
- ``task_callback`` (Crew level) — fires once when a Task completes.
  Receives the ``TaskOutput`` (description, summary, raw, json_dict,
  pydantic).

The adapter installs both callbacks on a Crew and translates them into
LoopGain observations. Either or both can be active depending on what the
user's loop looks like:

- Refinement *within* a single Task → use ``step_error_fn`` (per agent step).
- Sequential refinement *across* Tasks → use ``task_error_fn`` (per task).

Reference: https://docs.crewai.com/en/concepts/tasks
           https://docs.crewai.com/en/learn/sequential-process

Note: CrewAI does not expose a public way to *abort* a running Crew from a
callback. The adapter records observations and lets the Crew run to its
own completion; callers who need hard stops should set ``max_iter`` /
``max_execution_time`` on their Agents/Tasks. ``LoopGain.should_continue``
state is still honored on subsequent ``observe()`` calls (any further
callback firings after a terminal state are no-ops).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from loopgain.core import LoopGain


CrewStepFn = Callable[[Any], Optional[float]]
CrewTaskFn = Callable[[Any], Optional[float]]


class CrewAIAdapter:
    """Drive a CrewAI Crew with a LoopGain monitor.

    Args:
        lg: A ``LoopGain`` instance to drive.
        step_error_fn: Optional. Maps one agent step (an ``AgentAction`` /
            ``AgentFinish`` / similar object with ``thought``, ``tool``,
            ``result`` attributes or dict keys) to an error magnitude.
            Return ``None`` to skip the step.
        task_error_fn: Optional. Maps a ``TaskOutput`` to an error
            magnitude. Return ``None`` to skip.

    At least one of ``step_error_fn`` / ``task_error_fn`` must be provided
    so the adapter has something to observe.

    Example::

        from crewai import Agent, Crew, Task
        from loopgain import LoopGain
        from loopgain.integrations import CrewAIAdapter

        lg = LoopGain(target_error=0.1, max_iterations=20)
        adapter = CrewAIAdapter(
            lg=lg,
            task_error_fn=lambda output: count_failed_checks(output.raw),
        )
        crew = Crew(agents=[...], tasks=[...])
        adapter.install(crew)
        result = crew.kickoff()
    """

    framework_name = "crewai"

    def __init__(
        self,
        lg: "LoopGain",
        step_error_fn: Optional[CrewStepFn] = None,
        task_error_fn: Optional[CrewTaskFn] = None,
    ) -> None:
        if step_error_fn is None and task_error_fn is None:
            raise ValueError(
                "CrewAIAdapter requires step_error_fn or task_error_fn "
                "(or both) — at least one observation source"
            )
        self.lg = lg
        self.step_error_fn = step_error_fn
        self.task_error_fn = task_error_fn
        self._installed: list[tuple[Any, str, Any]] = []

    def install(self, crew: Any) -> None:
        """Install callbacks on the given Crew (or any object exposing
        ``step_callback`` / ``task_callback`` attributes).

        The previous values are saved and restored by ``uninstall()``.
        Calling ``install`` twice on the same Crew without an intervening
        ``uninstall`` overwrites the saved snapshot — only the most
        recently installed adapter can cleanly restore.
        """
        if self.step_error_fn is not None and hasattr(crew, "step_callback"):
            self._installed.append((crew, "step_callback", getattr(crew, "step_callback")))
            existing = getattr(crew, "step_callback")
            crew.step_callback = self._chain(existing, self._on_step)
        if self.task_error_fn is not None and hasattr(crew, "task_callback"):
            self._installed.append((crew, "task_callback", getattr(crew, "task_callback")))
            existing = getattr(crew, "task_callback")
            crew.task_callback = self._chain(existing, self._on_task)

    def uninstall(self) -> None:
        """Restore the original callbacks (if any) saved during install."""
        for target, attr, original in self._installed:
            try:
                setattr(target, attr, original)
            except Exception:
                pass
        self._installed.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.uninstall()

    def _on_step(self, step: Any) -> None:
        if self.step_error_fn is None or not self.lg.should_continue():
            return
        magnitude = self.step_error_fn(step)
        if magnitude is not None:
            self.lg.observe(magnitude, output=step)

    def _on_task(self, task_output: Any) -> None:
        if self.task_error_fn is None or not self.lg.should_continue():
            return
        magnitude = self.task_error_fn(task_output)
        if magnitude is not None:
            self.lg.observe(magnitude, output=task_output)

    @staticmethod
    def _chain(existing: Optional[Callable[[Any], None]], ours: Callable[[Any], None]) -> Callable[[Any], None]:
        """Compose with whatever callback the user already had installed,
        so the adapter doesn't clobber existing instrumentation."""
        if existing is None:
            return ours

        def chained(item: Any) -> None:
            try:
                existing(item)
            except Exception:
                # Don't let the user's callback break our observation pipeline.
                pass
            ours(item)

        return chained
