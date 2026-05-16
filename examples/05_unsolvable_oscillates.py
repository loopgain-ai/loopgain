"""Example 05 — Contradictory spec → OSCILLATING + best-so-far rollback.

The verifier checks three conditions:
  (a) the source must NOT contain bare keywords `async`/`await`, nor any
      of the coroutine drivers: `asyncio.run`, `run_until_complete`,
      `new_event_loop`, `get_event_loop`, `create_task`, `ensure_future`,
      `gather`.
  (b) the source MUST call `asyncio.sleep` somewhere.
  (c) the source MUST contain at least one coroutine driver from the
      list in (a) — otherwise calling `asyncio.sleep(...)` just produces
      an un-driven coroutine and the function never actually sleeps.

(a) forbids exactly the same set that (c) requires, so the minimum
achievable error is 1 — there's no way Claude can satisfy all three. The
loop swings between satisfying (a) and satisfying (c); error never reaches 0.

This is the **headline rollback demo**. LoopGain detects the Aβ ≈ 1
plateau, terminates cleanly, and `lg.result.best_output` is the
lowest-error iteration in the buffer — typically an earlier attempt,
not the terminal one.

target_error=None: error 0 is unreachable by construction, so there's
no target-met short-circuit to set; we rely entirely on stability
detection (OSCILLATING band) and max_iterations.

Expected band:  OSCILLATING with best_index < iterations_used - 1.
Loop type:      verify_revise.
"""

from __future__ import annotations

import re

from loopgain import LoopGain

from _common import (
    call_claude,
    get_client,
    print_comparison,
    print_iteration,
    print_result,
    print_rollback_note,
    send_telemetry,
)

WORKLOAD_ID = "example-05-unsolvable-oscillates"
FIXED_CAP = 10

STARTER = '''\
import time
import requests

def fetch(url):
    """Fetch a URL after a 1-second delay."""
    time.sleep(1)
    return requests.get(url).text
'''

SPEC = (
    "Rewrite the function below so that it sleeps via `asyncio.sleep(1)` "
    "instead of `time.sleep(1)`, AND the function must actually pause for "
    "one second when called (not just create an un-driven coroutine).\n\n"
    "HARD CONSTRAINTS — your source must NOT contain ANY of these tokens:\n"
    "  - the bare keywords `async` or `await`\n"
    "  - `asyncio.run`\n"
    "  - `run_until_complete`\n"
    "  - `new_event_loop`\n"
    "  - `get_event_loop`\n"
    "  - `create_task`\n"
    "  - `ensure_future`\n"
    "  - `gather`\n\n"
    "AND your source MUST call `asyncio.sleep(1)` somewhere AND must "
    "actually drive that coroutine so the function suspends.\n\n"
    "Return ONLY the Python source of the rewritten module. No prose, "
    "no fences, no comments.\n\n"
    f"Function to rewrite:\n```python\n{STARTER}```"
)

# Tokens forbidden by constraint (a) — these are exactly the coroutine
# drivers that constraint (c) requires, so satisfying both is impossible.
DRIVERS = [
    r"\basync\b", r"\bawait\b", r"asyncio\.run\b", r"run_until_complete",
    r"new_event_loop", r"get_event_loop", r"create_task", r"ensure_future",
    r"\bgather\b",
]
DRIVER_RE = re.compile("|".join(DRIVERS))


def strip_fences(text: str) -> str:
    m = re.match(r"^\s*```(?:python|py)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def evaluate(code: str):
    """Return (error, reason). 0 is unreachable by construction."""
    if not code:
        return 3, "no code returned"
    forbidden_hits = DRIVER_RE.findall(code)
    has_sleep = "asyncio.sleep" in code
    has_driver = bool(forbidden_hits)
    err = (1 if has_driver else 0) + (0 if has_sleep else 1) + (0 if has_driver else 1)
    reasons = []
    if has_driver:
        reasons.append(f"forbidden tokens used: {sorted(set(forbidden_hits))}")
    if not has_sleep:
        reasons.append("missing `asyncio.sleep` call")
    if not has_driver:
        reasons.append("no coroutine driver — `asyncio.sleep(...)` never runs")
    return err, "; ".join(reasons) or "satisfies all (unreachable)"


def one_iteration(client, prev_code: str, prev_reason: str):
    prompt = SPEC if not prev_code else (
        f"{SPEC}\n\nYour previous attempt was:\n```python\n{prev_code}\n```"
        f"\n\nVerifier reported: {prev_reason}\n\nTry again. Code only."
    )
    code = strip_fences(call_claude(client, prompt))
    err, reason = evaluate(code)
    return err, code, reason


def baseline_run(client):
    print(f"─── BASELINE: no LoopGain, fixed cap = {FIXED_CAP} ───")
    code, reason, err = "", "", -1
    for i in range(FIXED_CAP):
        err, code, reason = one_iteration(client, code, reason)
        print(f"  iter {i+1:>2}  error={err}  ({reason})")
    print(f"  → kept LAST output (terminal iter, error={err}).\n")
    return err, FIXED_CAP


def loopgain_run(client):
    print(f"─── WITH LOOPGAIN: target_error=None, max_iterations={FIXED_CAP} ───")
    lg = LoopGain(target_error=None, max_iterations=FIXED_CAP)
    code, reason = "", ""
    while lg.should_continue():
        err, code, reason = one_iteration(client, code, reason)
        first_line = code.splitlines()[0] if code else "[no code]"
        preview = f"err={err}; {reason}; {first_line}"
        state = lg.observe(err, output=code)
        print_iteration(lg.result.iterations_used, err, state, lg.eta, preview)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    print("Spec: rewrite to use asyncio.sleep without `async`/`await`.\n"
          "(Impossible by construction — expecting OSCILLATING + rollback.)\n")
    baseline_err, baseline_iters = baseline_run(client)
    lg = loopgain_run(client)
    print_comparison(baseline_iters, baseline_err, lg)
    print_rollback_note(lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="verify_revise")


if __name__ == "__main__":
    main()
