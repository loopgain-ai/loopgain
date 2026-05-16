# LoopGain examples — real Claude loops, baseline vs LoopGain

Seven runnable end-to-end scripts. Each one wraps a real Claude API loop
with a `LoopGain` monitor AND runs the same loop without LoopGain (fixed
iteration cap) so you can see the savings as the headline number.

| # | File | Pattern | Demonstrates |
|---|---|---|---|
| 01 | `01_code_pytest.py` | verify-revise | `TARGET_MET` on iter 1 — Codewars-grade problem with pytest as verifier |
| 02 | `02_json_extract.py` | verify-revise | `CONVERGING` / `TARGET_MET` — JSON extraction with schema validation |
| 03 | `03_essay_critique.py` | verify-revise (LLM-as-judge) | `STALLING` — rubric-loop plateau (the *Waste Report* case) |
| 04 | `04_sql_synth.py` | tool-use retry | Mixed bands — text-to-SQL with execution diff |
| 05 | `05_unsolvable_oscillates.py` | verify-revise | `OSCILLATING` + best-so-far rollback (headline demo) |
| 06 | `06_diverges.py` | refinement | `DIVERGING` + best-so-far rollback (headline demo) |
| 07 | `07_agentic_multistep.py` | multi-step reasoning | `TARGET_MET` — agentic goal pursuit across multi-dimensional constraints |

---

## Install

```bash
pip install 'loopgain[examples]'   # pulls in anthropic
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export LOOPGAIN_TELEMETRY_ENDPOINT="https://telemetry.loopgain.ai/v1/aggregate"
export LOOPGAIN_TELEMETRY_TOKEN="lgk_..."   # get one at https://loopgain.ai
```

Without `LOOPGAIN_TELEMETRY_TOKEN`, the loop still runs locally — only the
POST is skipped.

Override the model via `LOOPGAIN_EXAMPLE_MODEL` (default `claude-opus-4-7`).

---

## Run

```bash
python examples/01_code_pytest.py
```

Each script runs the SAME loop twice — once with a fixed iteration cap
(the universal `max_iterations=N` hack), once with LoopGain — and prints
a comparison block at the end. Sample output from example 01:

```
─── BASELINE: no LoopGain, fixed cap = 5 ───
  iter  1  error=  0  (always runs to cap)
  iter  2  error=  0  (always runs to cap)
  iter  3  error=  0  (always runs to cap)
  iter  4  error=  0  (always runs to cap)
  iter  5  error=  0  (always runs to cap)
  → kept LAST output. final error=0

─── WITH LOOPGAIN: target_error=0, max_iterations=5 ───
  iter  1  error=  0.00  state=TARGET_MET      eta=—

outcome:        converged
iterations:     1
best_error:     0.00
savings:        9

┌─ COMPARISON ──────────────────────────────────────────────────
│  Baseline:  5 iters, final error = 0, kept LAST
│  LoopGain:  1 iters, best  error = 0.0, kept best-so-far (iter 1) — state converged
│  Saved:     4 iterations (80%) of API spend
└──────────────────────────────────────────────────────────────

telemetry sent: True
```

The Saved line is the headline pilot-demo number: **measured savings,
not extrapolated**. Both runs make real Claude API calls.

For 05 (oscillating) and 06 (diverging) you'll also see a `rolled_back`
line — `best_index` is an earlier iteration than the terminal one, the
canonical "LoopGain rescued the output" punchline.

---

## Run all of them

```bash
make examples
```

(Manual only — there's no CI integration. Each invocation costs real API
budget; expect ~$0.10-$0.30 per example with Opus, roughly $1-3 total.)

---

## Watch the traces land in the dashboard

Open **[dashboard.loopgain.ai](https://dashboard.loopgain.ai)** and filter
by `workload_id` (`example-01-code-pytest`, `example-02-json-extract`, …)
or `loop_type` (`verify_revise`, `refinement`, `tool_use_retry`,
`multi_step_reasoning`).

---

## Privacy

These examples send aggregate telemetry only — Aβ statistics, state
transitions, iteration counts, library version, your opaque `workload_id`,
and a UTC hour-bucketed timestamp. **No prompts, no Claude completions,
no error contents, no per-iteration Aβ values are sent.** See
`loopgain/telemetry.py` for the exact payload shape; `tests/test_telemetry.py`
enforces the contract.

To keep everything local, self-host the
[receiver](https://github.com/loopgain-ai/telemetry-receiver) and point
`LOOPGAIN_TELEMETRY_ENDPOINT` at it.
