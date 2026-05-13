# LoopGain

**Barkhausen stability monitor for AI agent verify-revise loops.**

Replace `max_iterations=5` with a real-time loop-gain (`Aβ`) monitor that knows whether your agent loop is converging, stalling, oscillating, or diverging — and what to do in each case.

[![PyPI](https://img.shields.io/pypi/v/loopgain.svg)](https://pypi.org/project/loopgain/)
[![Python](https://img.shields.io/pypi/pyversions/loopgain.svg)](https://pypi.org/project/loopgain/)
[![npm](https://img.shields.io/npm/v/loopgain.svg)](https://www.npmjs.com/package/loopgain)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-73_passing-brightgreen.svg)](tests/)

Works with **LangGraph**, **CrewAI**, **AutoGen**, **Claude Agent SDK**, and any custom verify-revise loop that exposes an error signal. Pure Python, no runtime dependencies.

> **Note:** The npm package is currently a name reservation. The Python library is the working v0.1 release; a JavaScript port is planned for v0.2.

**Keywords:** AI agent loops · agentic AI · infinite loop detection · divergence detection · early stopping · convergence · agent orchestration · LLM stability · generator-verifier-reviser · feedback-loop control.

---

## Why

Production agent loops universally use `max_iterations=N` as their termination policy. It's the embarrassing default of agentic AI: you either waste compute (loop stops too late) or ship bad output (loop stops too early). LoopGain replaces it with a control-theoretic stability monitor based on the **Barkhausen criterion** — a foundational result from electrical-engineering feedback-oscillator analysis (1921).

The math is foundational. The product is the threshold bands, the best-so-far buffer, the ETA prediction, and the clean Python API.

---

## Install

```bash
pip install loopgain
```

Pure Python, no dependencies, supports Python 3.10+.

---

## Usage

Three lines of code wrap any verify-revise loop:

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

LoopGain measures empirical loop gain `Aβ = E(n) / E(n-1)` at every iteration. It smooths Aβ with a configurable EMA and classifies the result into five named bands:

| `Aβ_smooth` range | State | Action |
| --- | --- | --- |
| `< 0.3` | `FAST_CONVERGE` | Continue, predict ETA |
| `0.3 ≤ Aβ < 0.85` | `CONVERGING` | Continue, watch for upward drift |
| `0.85 ≤ Aβ < 0.95` | `STALLING` | Warn — diminishing returns |
| `0.95 ≤ Aβ ≤ 1.05` | `OSCILLATING` | Break — return best-so-far |
| `> 1.05` | `DIVERGING` | Abort — roll back to best-so-far |

Plus a short-circuit: if observed error drops at or below `target_error`, the loop stops immediately with state `TARGET_MET`.

The `±0.05` noise band around `Aβ=1` absorbs stochastic jitter from agent outputs without triggering false-positive aborts. The `0.85` `STALLING` boundary is an early warning — by the time `Aβ` crosses `1.0`, you've already wasted iterations.

These threshold defaults work well for typical agent loops out of the box. Tune them per domain (via the `ThresholdBands` argument) once you have production traces.

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

- `target_error` — Stop when an observed error drops at or below this. Default `0.0` means "never short-circuit on target met."
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

### `lg.send_telemetry(endpoint, token, workload_id=None, timeout=2.0) -> bool`

**Opt-in.** Send a single anonymized telemetry POST after the loop terminates. Best-effort — never raises, returns `True` on 2xx, `False` otherwise.

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

## Status

**v0.1.0 — initial public release.** Core library shipped. Framework adapters (LangGraph, CrewAI, AutoGen, Vesper) and the cloud-aggregator dashboard come in v0.2+. The math and the API surface are stable.

This is alpha software. The API may break before 1.0 if production usage surfaces design issues; pin the version.

---

## License

[Apache-2.0](LICENSE).

---

## Background

LoopGain applies the **Barkhausen stability criterion** (Heinrich Barkhausen, 1921 — the foundational result on when feedback amplifiers oscillate) to AI agent feedback loops. The criterion was originally a way to predict whether an electronic oscillator would sustain oscillation; it turns out to map cleanly onto any feedback loop you can attach an error signal to.

The cleanest summary: a verify-revise loop is a feedback system with measurable error magnitude. The ratio `E(n) / E(n-1)` is its empirical loop gain. The Barkhausen result tells you that loop gain less than 1 converges, equal to 1 oscillates, greater than 1 diverges. LoopGain operationalizes this: classifies the loop's current band, decides what to do, and tells you when you'll converge.

See [loopgain.ai](https://loopgain.ai) for the longer write-up.
