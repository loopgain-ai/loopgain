# LoopGain

**Barkhausen stability monitor for AI agent loops.**

Replace `max_iterations=5` with a real-time loop-gain (`Aβ`) monitor that knows whether your agent loop is converging, stalling, oscillating, or diverging — and what to do in each case.

[![PyPI](https://img.shields.io/pypi/v/loopgain.svg)](https://pypi.org/project/loopgain/)
[![Python](https://img.shields.io/pypi/pyversions/loopgain.svg)](https://pypi.org/project/loopgain/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-119_passing-brightgreen.svg)](tests/)

**Home:** [loopgain.ai](https://loopgain.ai)

Works for **any iterative AI workflow with a measurable error signal** — verify-revise loops, refinement passes, tool-use retry chains, RAG with self-correction, code-gen with linter feedback, multi-step reasoning loops. **Pre-built adapters for [LangGraph](#langgraph), [CrewAI](#crewai), [AutoGen](#autogen-v04), [LangChain](#langchain), [OpenAI Agents SDK](#openai-agents-sdk), and [Claude Agent SDK](#claude-agent-sdk)**; drop-in via the raw API for any custom stack. Pure Python, no runtime dependencies.

**Keywords:** AI agent loops · agentic AI · infinite loop detection · divergence detection · early stopping · convergence · agent orchestration · LLM stability · generator-verifier-reviser · feedback-loop control.

---

## Why

Production agent loops universally use `max_iterations=N` as their termination policy. It's the embarrassing default of agentic AI: you either waste compute (loop stops too late) or ship bad output (loop stops too early). LoopGain replaces it with a control-theoretic stability monitor based on the **Barkhausen criterion** — a foundational result from electrical-engineering feedback-oscillator analysis (1921).

---

## Install

```bash
pip install loopgain
```

Pure Python, no dependencies, supports Python 3.10+.

---

## Usage

Three lines of code wrap any iterative loop with a measurable error signal:

```python
from loopgain import LoopGain

lg = LoopGain(target_error=0.1)

while lg.should_continue():
    errors = verifier.verify(output)
    lg.observe(errors, output=output)
    output = reviser.revise(output, errors)

result = lg.result
print(result.outcome)              # "converged" | "oscillating" | "diverged" | "max_iterations"
print(result.best_output)          # the lowest-error iteration's output
print(result.iterations_used)
print(result.gain_margin)          # 1 / max(Aβ_smooth)
print(result.savings_vs_fixed_cap)
```

`observe()` accepts either a numeric error magnitude or any sequence (whose length becomes the magnitude). Pass `output=...` to enable the best-so-far buffer.

---

## How it works

LoopGain measures empirical loop gain at every iteration, then smooths it with an EMA:

```
Aβ(n)     = E(n) / E(n-1)
Aβ_smooth = EMA(Aβ, w=3)
```

It classifies `Aβ_smooth` into five named bands:

| `Aβ_smooth` range | State | Action |
| --- | --- | --- |
| `< 0.3` | `FAST_CONVERGE` | Continue, predict ETA |
| `0.3 ≤ Aβ < 0.85` | `CONVERGING` | Continue, watch for upward drift |
| `0.85 ≤ Aβ < 0.95` | `STALLING` | Warn — diminishing returns |
| `0.95 ≤ Aβ ≤ 1.05` | `OSCILLATING` | Break — return best-so-far |
| `> 1.05` | `DIVERGING` | Abort — roll back to best-so-far |

Plus a short-circuit: if observed error drops at or below `target_error`, the loop stops immediately with state `TARGET_MET`. The default `target_error=0.0` short-circuits on exactly zero error — the natural completion signal for verifier-driven loops. Pass `target_error=None` to disable the short-circuit and rely on stability detection alone.

The `±0.05` noise band around `Aβ=1` absorbs stochastic jitter from agent outputs without triggering false-positive aborts. The `0.85` `STALLING` boundary is an early warning — by the time `Aβ` crosses `1.0`, you've already wasted iterations.

These threshold defaults are derived from the Barkhausen-stability analysis and serve as reasonable starting points. Tune them per domain (via the `ThresholdBands` argument) once you have production traces.

---

## ETA prediction

When the loop is converging (`Aβ_smooth < 1`), LoopGain produces a closed-form prediction of iterations remaining:

```
n_remaining = log(E_target / E_current) / log(Aβ_smooth)
```

Available as `lg.eta` mid-loop. Returns `None` when the prediction isn't well-defined (no Aβ yet, target zero, or non-converging gain).

---

## Best-so-far rollback

LoopGain keeps a buffer of all observed outputs paired with their error scores. On termination it returns `argmin(error)`, not the last iteration:

| Terminal state | Returned output |
| --- | --- |
| `TARGET_MET` | Current output (by definition, the best) |
| `OSCILLATING` | Lowest-error iteration in the buffer |
| `DIVERGING` | Lowest-error iteration (which is *not* the last one) |

This transforms divergence detection from "abort with garbage" into "abort with the best you've seen so far" — a free quality floor.

---

## API reference

### `LoopGain(target_error=0.0, max_iterations=None, thresholds=None, smoothing_window=3, assumed_fixed_cap=10)`

Construct the monitor.

- `target_error` — Stop when an observed error drops at or below this. Default `0.0` short-circuits on exactly zero error (the natural completion signal for verifier-driven loops). Pass `None` to disable the short-circuit entirely.
- `max_iterations` — Hard safety cap. Default `None` (rely on stability detection). Recommended ~20–50 for production.
- `thresholds` — Custom `ThresholdBands` if defaults don't fit your domain.
- `smoothing_window` — EMA window for the smoothed Aβ. Default 3.
- `assumed_fixed_cap` — Used to compute `savings_vs_fixed_cap`. Default 10.

### `lg.observe(errors, output=None) -> str`

Record this iteration's errors and optional output. Returns the current state name. `errors` accepts a number (used directly) or any sequence (length used as magnitude).

### `lg.should_continue() -> bool`

Returns `False` once a terminal state fires.

### `lg.state -> str`

Current state name. One of `INIT`, `FAST_CONVERGE`, `CONVERGING`, `STALLING`, `OSCILLATING`, `DIVERGING`, `TARGET_MET`, `MAX_ITERATIONS`.

### `lg.eta -> int | None`

Predicted iterations to reach target. `None` when not well-defined.

### `lg.gain_margin -> float | None`

`1 / max(Aβ_smooth)`. `> 1` means stable headroom across the entire run.

### `lg.result -> LoopGainResult`

Terminal result with `outcome`, `iterations_used`, `best_index`, `best_output`, `best_error`, `convergence_profile`, `error_history`, `gain_margin`, `savings_vs_fixed_cap`. Safe to call mid-loop.

### `lg.send_telemetry(endpoint, token, workload_id=None, timeout=2.0, allow_insecure=False, framework=None, loop_type=None, team=None, include_per_iteration=True) -> bool`

**Opt-in.** Send a single anonymized telemetry POST after the loop terminates. Best-effort — never raises, returns `True` on 2xx, `False` otherwise. Adapters auto-stamp `framework`; `loop_type` and `team` are free-form labels that surface as filters in the dashboard. Pass `include_per_iteration=False` to send aggregate summary only.

```python
import os
from loopgain import LoopGain

lg = LoopGain(target_error=0.1)
# ... run the loop ...
lg.send_telemetry(
    endpoint=os.environ["LOOPGAIN_TELEMETRY_ENDPOINT"],   # or hardcode
    token=os.environ["LOOPGAIN_TELEMETRY_TOKEN"],         # never hardcode
    workload_id="my-rag-pipeline",                        # opaque label
)
```

Recommended setup: store the token outside source. Two clean options:

```bash
# Option A: environment variable (simplest)
export LOOPGAIN_TELEMETRY_ENDPOINT="https://telemetry.loopgain.ai/v1/aggregate"
export LOOPGAIN_TELEMETRY_TOKEN="lgk_..."   # add to ~/.zshrc or ~/.bashrc

# Option B: macOS Keychain (more secure)
pip install keyring
python3 -c "import keyring; keyring.set_password('loopgain', 'telemetry', input('Token: '))"
# Then in code: keyring.get_password('loopgain', 'telemetry')
```

What is sent: state transitions, Aβ summary (min/max/median), gain margin, rollback flag, iterations used, savings, library version, optional opaque `workload_id`, threshold config, hour-bucketed timestamp.

**What is NEVER sent: prompts, completions, error contents, output buffer, individual Aβ values, or any customer identity beyond the bearer token.** Privacy contract is enforced by the payload-shape unit tests in `tests/test_telemetry.py`.

The hosted endpoint at `telemetry.loopgain.ai` is one acceptable destination. The [receiver](https://github.com/loopgain-ai/telemetry-receiver) and [dashboard](https://github.com/loopgain-ai/dashboard) are both open-source — self-host to keep telemetry fully under your control.

---

## Framework adapters

Thin wrappers under `loopgain.integrations` drive each major agent framework's iteration with a `LoopGain` monitor and auto-stamp `framework="<name>"` on telemetry. The frameworks themselves are **optional dependencies** — install the extra you need:

```bash
pip install 'loopgain[langgraph]'          # LangGraph
pip install 'loopgain[crewai]'             # CrewAI
pip install 'loopgain[autogen]'            # AutoGen v0.4+
pip install 'loopgain[langchain]'          # LangChain (create_agent / AgentExecutor)
pip install 'loopgain[openai-agents]'      # OpenAI Agents SDK
pip install 'loopgain[claude-agent-sdk]'   # Anthropic Claude Agent SDK
pip install 'loopgain[all]'                # all six
```

All adapters take a `LoopGain` instance plus an `error_fn` you provide — the framework doesn't know what your error signal is, so the adapter doesn't either. `error_fn` returns a non-negative number (or `None` to skip an iteration).

### LangGraph

Drives `graph.stream(input, stream_mode="updates")`. Each update is one iteration.

```python
from loopgain import LoopGain
from loopgain.integrations import LangGraphAdapter

graph = build_my_verify_revise_graph().compile()
lg = LoopGain(target_error=0.1, max_iterations=20)

adapter = LangGraphAdapter(
    lg=lg,
    error_fn=lambda update: len(update.get("verifier", {}).get("errors", [])),
)
final_state = adapter.run(graph, {"draft": initial})

lg.send_telemetry(
    endpoint=os.environ["LOOPGAIN_TELEMETRY_ENDPOINT"],
    token=os.environ["LOOPGAIN_TELEMETRY_TOKEN"],
    workload_id="rag-rewrite",
    framework=adapter.framework_name,        # "langgraph", auto-stamped
)
```

`adapter.stream(...)` yields each item if you want the full trace; `adapter.arun(...)` / `adapter.astream(...)` are the async counterparts and accept an async `error_fn`.

### CrewAI

Installs `step_callback` and/or `task_callback` on a Crew. Pick whichever granularity matches your loop — `step_error_fn` for refinement *within* a Task, `task_error_fn` for refinement *across* Tasks.

```python
from crewai import Crew
from loopgain import LoopGain
from loopgain.integrations import CrewAIAdapter

lg = LoopGain(target_error=0.1, max_iterations=20)
adapter = CrewAIAdapter(
    lg=lg,
    task_error_fn=lambda task_output: count_failed_checks(task_output.raw),
)
crew = Crew(agents=[...], tasks=[...])
adapter.install(crew)
result = crew.kickoff()
adapter.uninstall()         # or use `with CrewAIAdapter(...) as a:` context

lg.send_telemetry(
    endpoint=...,
    token=...,
    framework=adapter.framework_name,        # "crewai"
)
```

The adapter chains with any callback you already had installed — your existing instrumentation isn't overwritten.

### AutoGen (v0.4+)

Wraps `team.run_stream(task=...)`. In a verify-revise rotation, filter to the verifier's messages with `observe_sources={"verifier"}` so only it drives `observe()`.

```python
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
result = await adapter.run(team, task="...")

lg.send_telemetry(
    endpoint=...,
    token=...,
    framework=adapter.framework_name,        # "autogen"
)
```

Pass a `cancellation_token` to `adapter.run(...)` and the adapter will cancel it when LoopGain reaches a terminal state (target met, oscillation, divergence). The legacy v0.2 `ConversableAgent.initiate_chat` API is **not** supported — use the v0.4 event-driven runtime.

### LangChain

Duck-types against any LangChain agent that exposes `.stream(input, **kwargs)` / `.astream(input, **kwargs)` — both the current `langchain.agents.create_agent()` (v1+) and the legacy `AgentExecutor`. The adapter forwards `**stream_kwargs` verbatim, so the chunk shape your `error_fn` sees is the one your agent emits.

```python
from langchain.agents import create_agent
from loopgain import LoopGain
from loopgain.integrations import LangChainAdapter

agent = create_agent(model="gpt-5-nano", tools=[get_weather])
lg = LoopGain(target_error=0.0, max_iterations=20)

def error_fn(chunk):
    if chunk.get("type") != "updates":
        return None
    # Count unresolved tool calls; drops to 0 once the agent stops calling tools.
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

lg.send_telemetry(
    endpoint=...,
    token=...,
    framework=adapter.framework_name,        # "langchain"
)
```

For legacy `AgentExecutor`: just drop the `stream_mode` / `version` kwargs; each yielded chunk is an `AddableDict` per step (parse `intermediate_steps` or the terminal `output` key in your `error_fn`).

### OpenAI Agents SDK

Wraps `Runner.run_streamed(agent, input).stream_events()`. The SDK is async-first; the adapter mirrors that. A `run_sync` helper wraps the async path with `asyncio.run` for synchronous callers.

```python
from agents import Agent, function_tool
from loopgain import LoopGain
from loopgain.integrations import OpenAIAgentsAdapter

agent = Agent(name="Reviser", instructions="...", tools=[...])

lg = LoopGain(target_error=0.0, max_iterations=20)

def error_fn(event):
    # Default observes only run_item_stream_event; pull the verifier's
    # reported failure count off tool outputs.
    if event.item.type == "tool_call_output_item":
        return float(event.item.output.get("failures", 0))
    return None

adapter = OpenAIAgentsAdapter(lg=lg, error_fn=error_fn)
result = await adapter.run(agent, input="Fix the bug.")
print(result.final_output)

lg.send_telemetry(
    endpoint=...,
    token=...,
    framework=adapter.framework_name,        # "openai-agents"
)
```

By default the adapter only forwards `run_item_stream_event` to `error_fn` — pass `observe_event_types=None` to see every event (including raw token deltas and agent-handoff notifications). When LoopGain reaches a terminal state, the adapter best-effort calls `.cancel()` on the underlying `RunResultStreaming`.

### Claude Agent SDK

Wraps Anthropic's `claude_agent_sdk.query(prompt=..., options=...)` async iterator. By default observes only `AssistantMessage` (skips `UserMessage` / `SystemMessage` / `ResultMessage`); override with `observe_message_types=None` or a custom tuple.

```python
from claude_agent_sdk import ClaudeAgentOptions, TextBlock
from loopgain import LoopGain
from loopgain.integrations import ClaudeAgentSDKAdapter

def error_fn(message):
    # Count `FAIL:` markers a self-verifying persona emits.
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

lg.send_telemetry(
    endpoint=...,
    token=...,
    framework=adapter.framework_name,        # "claude-agent-sdk"
)
```

For the bidirectional `ClaudeSDKClient` use case, pass `message_iterator=client.receive_messages()` instead of `prompt=...`.

### Custom integrations

For frameworks without an adapter, the raw `LoopGain.observe()` API works against any iterable. The adapters are 100-200 lines each — copy one of `loopgain/integrations/{langgraph,crewai,autogen,langchain,openai_agents,claude_agent_sdk}.py` as a starting point.

---

## Status

**Initial public release.** Core library shipped (current version: see the PyPI badge at the top). Framework adapters (LangGraph, CrewAI, AutoGen, LangChain, OpenAI Agents SDK, Claude Agent SDK) are installable as optional extras. The cloud-aggregator [telemetry receiver](https://github.com/loopgain-ai/telemetry-receiver) and [dashboard](https://github.com/loopgain-ai/dashboard) are live as separate open-source repos. The math and the API surface are stable.

This is alpha software. The API may break before 1.0 if production usage surfaces design issues; pin the version.

---

## License

[Apache-2.0](LICENSE).

---

## Background

LoopGain applies the **Barkhausen stability criterion** (Heinrich Barkhausen, 1921 — the foundational result on when feedback amplifiers oscillate) to AI agent feedback loops. The criterion was originally a way to predict whether an electronic oscillator would sustain oscillation; it turns out to map cleanly onto any feedback loop you can attach an error signal to.

The cleanest summary: an iterative AI loop with a measurable error signal is a feedback system. The ratio `E(n) / E(n-1)` is its empirical loop gain. The Barkhausen result tells you that loop gain less than 1 converges, equal to 1 oscillates, greater than 1 diverges. LoopGain operationalizes this: classifies the loop's current band, decides what to do, and tells you when you'll converge.

Loop types this applies to in practice:

- **Verify-revise loops** (GVR pattern) — generator produces, verifier finds issues, reviser fixes. Error = issue count or severity-weighted score.
- **Refinement loops** — initial output, iterate to improve. Error = distance from target spec / rubric score.
- **Tool-use retry chains** — agent calls tool, gets back error/success, retries. Error = consecutive failure count or aggregate score.
- **RAG with self-correction** — retrieve, generate, critique, re-retrieve. Error = critique severity or hallucination score.
- **Code generation with linter/test feedback** — generate, run tests/linter, fix, repeat. Error = failing test count or linter violation count.
- **Multi-step reasoning loops** — ReAct-style think/act/observe iterations. Error = whatever the agent's quality assessor returns.
- **Custom feedback loops** — anything where you can produce a number that should drop toward zero as the loop succeeds.
