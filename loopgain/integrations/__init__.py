"""Framework integration adapters for LoopGain.

Each adapter is a thin wrapper that drives the host framework's iteration,
calls ``LoopGain.observe()`` on each step with an error magnitude derived
from a user-provided ``error_fn``, and (optionally) sends telemetry on
completion with ``framework="<name>"`` auto-stamped.

Adapters are isolated submodules so the host frameworks (langgraph, crewai,
autogen) remain *optional* dependencies. Importing this package does not
import any framework — each adapter only imports its framework when its
class is instantiated, and surfaces a clear ``ImportError`` if missing.

Install adapter extras::

    pip install 'loopgain[langgraph]'   # LangGraph
    pip install 'loopgain[crewai]'      # CrewAI
    pip install 'loopgain[autogen]'     # AutoGen v0.4+
    pip install 'loopgain[all]'         # all of the above

Common pattern::

    from loopgain import LoopGain
    from loopgain.integrations import LangGraphAdapter   # or CrewAIAdapter, AutoGenAdapter

    lg = LoopGain(target_error=0.1, max_iterations=20)
    adapter = LangGraphAdapter(
        lg=lg,
        error_fn=lambda update: len(update.get("verifier_errors") or []),
    )
    final_state = adapter.run(graph, input_state)

    # Optional: ship telemetry with framework auto-stamped.
    lg.send_telemetry(
        endpoint="https://telemetry.loopgain.ai/v1/aggregate",
        token=os.environ["LOOPGAIN_TELEMETRY_TOKEN"],
        workload_id="rag-rewrite",
        framework=adapter.framework_name,   # "langgraph"
    )
"""

from __future__ import annotations

# Adapters are imported lazily so importing this package does NOT pull in
# langgraph / crewai / autogen. Each name resolves on first attribute access
# and surfaces a clear ImportError if its host framework isn't installed.
__all__ = [
    "LangGraphAdapter",
    "CrewAIAdapter",
    "AutoGenAdapter",
]


def __getattr__(name: str):
    if name == "LangGraphAdapter":
        from loopgain.integrations.langgraph import LangGraphAdapter

        return LangGraphAdapter
    if name == "CrewAIAdapter":
        from loopgain.integrations.crewai import CrewAIAdapter

        return CrewAIAdapter
    if name == "AutoGenAdapter":
        from loopgain.integrations.autogen import AutoGenAdapter

        return AutoGenAdapter
    raise AttributeError(f"module 'loopgain.integrations' has no attribute {name!r}")
