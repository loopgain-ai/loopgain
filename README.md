# LoopGain

**An open-source cost controller for AI agent loops.**

AI agent loops waste time and money when they don't know when to stop. LoopGain measures the loop in real time and stops it the moment it has actually converged — and rolls back before it degrades — instead of running to a fixed `max_iterations` cap.

> **Benchmark — 2,000 paired trials across 10 workload cells** ([run it yourself](https://github.com/loopgain-ai/loopgain-bench)):
>
> - **92.8% less API spend** than `max_iter=20` — $27.05 → $1.94 in total benchmark spend
> - **~15× faster** — median wall-clock per trial 30.9s → 2.1s
> - **Quality preserved, not traded for speed** — judge win-rate 0.50–0.63 on natural-distribution workloads (W1–W4, CI excluding null on most cells), 0.92–0.95 on engineered-failure workloads (W5); 0.678 weighted preference across 1,800 judge comparisons
> - **Zero of six kill criteria fired** (all six pre-registered with thresholds before the run)

**Honest limits, up front:** LoopGain detects *convergence, not correctness* — it knows when more iterations won't help, not whether the answer is right, and it's only as good as the verifier behind your error signal. [The full list of what it can't do →](#what-loopgain-does-and-doesnt-guarantee)

[![PyPI](https://img.shields.io/pypi/v/loopgain.svg)](https://pypi.org/project/loopgain/)
[![Python](https://img.shields.io/pypi/pyversions/loopgain.svg)](https://pypi.org/project/loopgain/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-190%2B_passing-brightgreen.svg)](tests/)

**Home:** [loopgain.ai](https://loopgain.ai)

Works for **any iterative AI workflow with a measurable error signal** — verify-revise loops, refinement passes, tool-use retry chains, RAG with self-correction, code-gen with linter feedback, multi-step reasoning loops. **Pre-built adapters for [LangGraph](#langgraph), [CrewAI](#crewai), [AutoGen](#autogen-v04), [LangChain](#langchain), [OpenAI Agents SDK](#openai-agents-sdk), and [Claude Agent SDK](#claude-agent-sdk)**; drop-in via the raw API for any custom stack. Pure Python, no runtime dependencies.

---

## Why

Production agent loops universally use `max_iterations=N` as their termination policy. It's the embarrassing default of agentic AI: you either waste compute (loop stops too late) or ship bad output (loop stops too early). LoopGain replaces it with a control-theoretic stop-and-rollback policy grounded in the **Barkhausen criterion** — a foundational result from electrical-engineering feedback-oscillator analysis (1921).

---

## Install

```bash
pip install loopgain
```

Pure Python, no dependencies, supports Python 3.10+.

**Using Claude Code?** The [loopgain-plugin](https://github.com/loopgain-ai/loopgain-plugin)
scans your whole repo for wrappable loops — literal, recursive, graph-cycle, and
semantic — and proposes reviewed diffs one file at a time (never auto-applied):
```
/plugin marketplace add loopgain-ai/loopgain-plugin
/plugin install loopgain
```

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
print(result.outcome)              # "converged" | "oscillating" | "diverged" | "stalled" | "max_iterations"
print(result.best_output)          # the lowest-error iteration's output
print(result.iterations_used)
print(result.savings_vs_fixed_cap)
```

`observe()` accepts either a numeric error magnitude or any sequence (whose length becomes the magnitude). Pass `output=...` to enable the best-so-far buffer.

---

## Defining your error signal

The one thing you provide is the **error signal**: a single non-negative number, every iteration, that says how wrong the current output is. **Lower is better; zero means done.** LoopGain doesn't know what your loop does — it just watches that number's trajectory and decides whether to keep going, stop, or roll back.

Your loop already has some way of knowing the output isn't good yet (or it wouldn't keep revising). Turn that into a number:

| Loop | Error signal = |
| --- | --- |
| Agentic coding (write code → run tests) | number of **failing tests** (10 → 3 → 0) |
| JSON / structured extraction | number of **schema violations** |
| RAG with self-correction | number of **required facts still missing** |
| Self-refinement with an LLM judge | judge's **gap to target** (e.g. `10 − quality_score`) |
| Lint / format loop | **lint error count** |

The only rules: non-negative, and **smaller as the output gets better**. Returning the raw list of problems works directly — `observe()` uses its length as the magnitude (e.g. hand it the list of failing tests).

If your quality is fuzzy and has no natural "zero," run with `target_error=None`: LoopGain then stops when the number **stops improving**, wherever that plateau is, instead of waiting for an exact target.

Every stop/continue decision is made from this one number, so **LoopGain is only as good as the error signal you give it** — pick one that genuinely tracks output quality.

---

## How it works

LoopGain measures empirical loop gain (`Aβ = E(n) / E(n-1)`) at every iteration and exposes it as a smoothed time series for visualization. The decision engine, however, classifies the **full error trajectory** using four features:

```
E_ratio   = E_current / E_first      # cumulative reduction
slope_log = OLS slope of log10(E)    # geometric trend direction
slope_p   = t-test p-value of slope  # statistical significance
osc_std   = std of detrended log10(E) # oscillation magnitude
```

It routes the trajectory into one of five named states:

| State | Condition | Action |
| --- | --- | --- |
| `FAST_CONVERGE` | cumulative reduction to ≤ 10% of E_first | Continue |
| `CONVERGING` | negative slope with `p < 0.05`, OR cumulative ≤ 50% | Continue, watch for upward drift |
| `STALLING` | no significant slope, no detectable oscillation | Stop after 2 consecutive readings — return best-so-far |
| `OSCILLATING` | high residual variance with flat trend | Stop — return best-so-far |
| `DIVERGING` | positive slope with `p < 0.05` AND cumulative > 110% | Abort — roll back to best-so-far |

Plus a short-circuit: if observed error drops at or below `target_error`, the loop stops immediately with state `TARGET_MET`. The default `target_error=0.0` short-circuits on exactly zero error — the natural completion signal for verifier-driven loops. Pass `target_error=None` to disable the short-circuit and rely on stability detection alone.

The decision is **conservative by design**: requiring both statistical significance and meaningful cumulative motion before terminating prevents false-positive aborts on noisy real-LLM error series. Validated at 98.8% macro-averaged accuracy across 5 regimes on N=1000 deterministic-mock trajectories (see `RESULTS_v2_classifier.md`). The STALLING ceiling of ~94% is the t-test's irreducible 5% type-I error rate, not a classifier weakness.

**Recommended minimum: 6 iterations** for reliable trend significance. At n≤4 the t-test is severely underpowered (df=2 requires |t|>4.3 for p<0.05) — the classifier conservatively falls back to STALLING when evidence is thin. The thresholds are derived analytically (control theory + statistical convention), not fitted; tune them per domain via the `TrajectoryThresholds` argument once you have production traces.

**Legacy single-feature classifier:** the original v0.1 single-Aβ-band classifier (thresholds 0.3 / 0.85 / 0.95 / 1.05) is still available via `LoopGain(classifier='legacy_bands')` for callers that have empirically tuned the bands to a specific workload.

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

## See it across a fleet (optional dashboard)

The library is the whole product locally — telemetry is opt-in and self-hostable. If you want a fleet view of every loop's stability, cost, and rollbacks across a team, there's a hosted dashboard fed by the [telemetry receiver](https://github.com/loopgain-ai/telemetry-receiver):

[![LoopGain dashboard — loop health, convergence, waste, and rollbacks across a fleet](https://loopgain.ai/dashboard-demo.png)](https://dashboard.loopgain.ai/demo)

**[Open the live demo →](https://dashboard.loopgain.ai/demo)** — no signup, real benchmark data.

The receiver and dashboard are both open-source — self-host to keep telemetry entirely under your control.

### Repositories

| Repo | What it is |
| --- | --- |
| [**loopgain**](https://github.com/loopgain-ai/loopgain) | This library — the Apache-2.0 control loop (you are here) |
| [**telemetry-receiver**](https://github.com/loopgain-ai/telemetry-receiver) | Cloudflare Worker that ingests anonymized loop telemetry |
| [**dashboard**](https://github.com/loopgain-ai/dashboard) | The fleet dashboard — self-hostable |
| [**loopgain-bench**](https://github.com/loopgain-ai/loopgain-bench) | The reproducible 2,000-trial benchmark behind the numbers above |
| [**loopgain-plugin**](https://github.com/loopgain-ai/loopgain-plugin) | Claude Code plugin — scans a repo for wrappable loops, proposes reviewed diffs |

---

## What LoopGain does and doesn't guarantee

LoopGain saves money by stopping a loop once it stops improving — fewer iterations, fewer tokens. In our [public benchmark](https://github.com/loopgain-ai/loopgain-bench), that was a **92.8% cut in total API spend** vs `max_iterations=20`, with output quality preserved. Two honest limits:

- **Savings depend on your workload.** Loops that usually succeed fast save the most (~96%); adversarial, failure-prone loops save less (~78–84%). The headline is a blend — run the benchmark on your own loops before quoting a number.
- **LoopGain detects convergence, not correctness.** It stops when your error signal stops improving — which means more iterations won't help, *not* that the loop succeeded. On the benchmark this preserved quality (it rarely stopped early on a worse output; false-stop rate ≤4.5%), but a loop can stall with the error still above zero — a plateau at, say, 2 failing tests. So check `result.best_error` (or your own pass/fail) before you trust the output: if it plateaued short of your target, that's a quality gap LoopGain can't see, and a false stop that forces a rerun is the one way it eats into the savings. LoopGain decides *when to stop*; you decide *whether the answer is good enough*.
- **LoopGain is only as right as your verifier.** It acts on the error signal you give it. If your verifier reports zero errors, LoopGain trusts that and stops — so a verifier with blind spots can report success on an answer that is still wrong, and LoopGain will confidently stop there. This is not the plateau case above: the error reads zero and the loop looks like a clean success, so neither LoopGain nor its convergence signal can flag it. The quality of the stop is bounded by the quality of the check behind your error signal. We measured this on the benchmark's code-gen workload: **4.5% of converged runs (16/355) passed every check the loop ran but failed the full held-out test suite** — and that's a floor, not a ceiling, because the in-loop verifier there was strong; a weaker verifier exposes more. (Distinct from the ≤4.5% false-stop rate above — the numbers coincide, the failure modes don't.) Pair LoopGain with the strongest verifier you can afford at the stop — executable tests over a sampled subset, a schema or type check over a vibe, a held-out check the loop didn't optimize against. **[How to design a strong verifier](https://loopgain.ai/blog/posts/how-to-design-a-strong-verifier/)** is a field guide to exactly this.

---

## API reference

### `LoopGain(target_error=0.0, max_iterations=50, thresholds=None, trajectory_thresholds=None, classifier='trajectory', smoothing_window=3, assumed_fixed_cap=10)`

Construct the monitor.

- `target_error` — Stop when an observed error drops at or below this. Default `0.0` short-circuits on exactly zero error (the natural completion signal for verifier-driven loops). Pass `None` to disable the short-circuit entirely.
- `max_iterations` — Hard safety backstop. Default `50` so the loop can never run unbounded; a stability verdict normally terminates it well before this. Pass `None` to opt into a fully unbounded loop (only safe if your loop is guaranteed to reach `target_error` or a stop-state), or a smaller integer to cap tighter.
- `thresholds` — Custom `ThresholdBands` for the legacy single-Aβ-band classifier. Ignored when `classifier='trajectory'`.
- `trajectory_thresholds` — Custom `TrajectoryThresholds` for the multi-feature classifier (the default). Override only with workload-specific evidence.
- `classifier` — `'trajectory'` (default, v0.2 multi-feature classifier) or `'legacy_bands'` (v0.1 single-Aβ-band classifier).
- `smoothing_window` — EMA window for the smoothed Aβ series (always maintained for visualization, regardless of classifier choice). Default 3.
- `assumed_fixed_cap` — Used to compute `savings_vs_fixed_cap`. Default 10.

### `lg.observe(errors, output=None) -> str`

Record this iteration's errors and optional output. Returns the current state name. `errors` accepts a number (used directly) or any sequence (length used as magnitude).

### `lg.should_continue() -> bool`

Returns `False` once a terminal state fires.

### `lg.state -> str`

Current state name. One of `INIT`, `FAST_CONVERGE`, `CONVERGING`, `STALLING`, `OSCILLATING`, `DIVERGING`, `TARGET_MET`, `MAX_ITERATIONS`. The corresponding terminal `result.outcome` values are `converged`, `oscillating`, `diverged`, `stalled` (v0.2 trajectory mode only — STALLING terminating after 2 consecutive readings), `max_iterations`, or `in_progress`.

### `lg.result -> LoopGainResult`

Terminal result with `outcome`, `iterations_used`, `best_index`, `best_output`, `best_error`, `convergence_profile`, `error_history`, `savings_vs_fixed_cap`. Safe to call mid-loop.

### `lg.send_telemetry(endpoint=None, token=None, workload_id=None, timeout=2.0, allow_insecure=False, framework=None, loop_type=None, team=None, include_per_iteration=True, retries=2, retry_backoff=0.25, actual_dollars_spent=None, actual_dollars_saved=None) -> bool`

**Opt-in.** Send a single anonymized telemetry POST after the loop terminates. Best-effort — never raises, returns `True` on 2xx, `False` otherwise. Adapters auto-stamp `framework`; `loop_type` and `team` are free-form labels that surface as filters in the dashboard. Pass `include_per_iteration=False` to send aggregate summary only.

`endpoint` and `token` are optional (v0.6.3+): with `LOOPGAIN_TELEMETRY_ENDPOINT` and `LOOPGAIN_TELEMETRY_TOKEN` exported, a bare `lg.send_telemetry()` is fully configured — the endpoint may be the receiver base URL (`https://telemetry.loopgain.ai`) or the full `/v1/aggregate` path. Nothing configured → returns `False`, sends nothing.

`actual_dollars_spent` and `actual_dollars_saved` are optional real-cost fields (v0.6.1+). Populate them only when you have a genuinely *measured* dollar figure — summed real API usage x list price, or an actually-executed paired-baseline comparison run. Never a formula-derived estimate. When populated, the dashboard displays your real number instead of its iter-count x $/iter extrapolation; passing an estimate through this field would present it as ground truth to every consumer of your tenant's data, not just you.

```python
from loopgain import LoopGain

lg = LoopGain(target_error=0.1)
# ... run the loop ...
lg.send_telemetry(workload_id="my-rag-pipeline")  # endpoint/token from env (v0.6.3+)
```

Verify the pipeline before wiring a real loop — `loopgain doctor` runs a tiny in-process loop (no model calls, $0) and sends one test event:

```bash
export LOOPGAIN_TELEMETRY_ENDPOINT="https://telemetry.loopgain.ai"
export LOOPGAIN_TELEMETRY_TOKEN="lgk_..."
loopgain doctor
# -> event accepted by the receiver -> appears in your dashboard as 'loopgain-doctor'
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

What is sent: state transitions, Aβ summary (min/max/median), rollback flag, iterations used, savings, library version, optional opaque `workload_id`, threshold config, hour-bucketed timestamp — and, unless you pass `include_per_iteration=False`, a length-capped per-iteration trajectory (smoothed Aβ values and numeric error magnitudes; this is what drives the dashboard's convergence-profile scrubbing).

**What is NEVER sent: prompts, completions, error contents, the output buffer, or any customer identity beyond the bearer token.** Numeric error *magnitudes* are sent (they're the loop-gain signal); error *contents* never are. Privacy contract is enforced by the payload-shape unit tests in `tests/test_telemetry.py`.

The hosted endpoint at `telemetry.loopgain.ai` is one acceptable destination. The [receiver](https://github.com/loopgain-ai/telemetry-receiver) and [dashboard](https://github.com/loopgain-ai/dashboard) are both open-source — self-host to keep telemetry fully under your control.

> **This is not the same as anonymous usage telemetry.** `send_telemetry` sends *your* loop data to *your* dashboard, and only when you call it. There's a separate, opt-in **funnel** telemetry described below. The two never share data or code.

---

## Anonymous funnel telemetry (opt-in, off by default)

LoopGain can report **anonymous usage counts** so a solo maintainer can tell whether the library is actually being used — install → first `observe()` → recurring use. **It is opt-in and default-decline: nothing is sent unless you explicitly turn it on.**

```bash
loopgain telemetry --show       # status + exactly what would be sent
loopgain telemetry --enable     # opt in   (or: export LOOPGAIN_TELEMETRY=1)
loopgain telemetry --disable    # opt out  (or: export LOOPGAIN_TELEMETRY=0)
```

`DO_NOT_TRACK=1` is honored as a hard opt-out, and CI environments are auto-detected and declined silently. When enabled, payloads carry only a locally-generated random id (not derived from your machine), hour-bucketed timestamps, library/Python/OS versions, the adapter in use, and a coarse outcome count. **Prompts, outputs, error contents, keys, paths, and IPs are never collected.** Delivery is batched, async, https-only, and fail-silent — it can never break your loop. Full details and the privacy contract: **[TELEMETRY.md](TELEMETRY.md)**.

If LoopGain is useful to you, opting in is the cheapest way to support the project — these counts are the only signal a solo-maintained library has that it's working for anyone.

---

## Command-line interface

```bash
loopgain --version              # or: loopgain version
loopgain telemetry --show       # inspect / control anonymous funnel telemetry
python -m loopgain telemetry --show   # equivalent, without the console script
```

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

The cleanest summary: an iterative AI loop with a measurable error signal is a feedback system. The ratio `E(n) / E(n-1)` is its empirical loop gain. The Barkhausen result tells you that loop gain less than 1 converges, equal to 1 oscillates, greater than 1 diverges. LoopGain operationalizes this: classifies the loop's current band, and decides what to do — stop, continue, or roll back to the best output seen so far.

Loop types this applies to in practice:

- **Verify-revise loops** (GVR pattern) — generator produces, verifier finds issues, reviser fixes. Error = issue count or severity-weighted score.
- **Refinement loops** — initial output, iterate to improve. Error = distance from target spec / rubric score.
- **Tool-use retry chains** — agent calls tool, gets back error/success, retries. Error = consecutive failure count or aggregate score.
- **RAG with self-correction** — retrieve, generate, critique, re-retrieve. Error = critique severity or hallucination score.
- **Code generation with linter/test feedback** — generate, run tests/linter, fix, repeat. Error = failing test count or linter violation count.
- **Multi-step reasoning loops** — ReAct-style think/act/observe iterations. Error = whatever the agent's quality assessor returns.
- **Custom feedback loops** — anything where you can produce a number that should drop toward zero as the loop succeeds.
