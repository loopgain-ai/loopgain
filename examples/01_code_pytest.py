"""Example 01 — Code generation with pytest feedback.

Real verify-revise loop on real Claude. Each iteration:
  1. Claude writes (or revises) a Python module to satisfy a spec.
  2. We run pytest against a fixed test file and count failing tests.
  3. The failing-test count is the error signal.

This example runs the SAME loop twice so you can see what LoopGain
actually buys you:

  BASELINE — the "max_iterations=N" hack. Runs a fixed cap of iterations
             regardless of whether the loop already succeeded.
  LOOPGAIN — short-circuits on success (target_error=0), monitors Aβ for
             stability, returns best-so-far on terminal states.

Expected band:  FAST_CONVERGE → TARGET_MET on a solvable problem.
Loop type:      verify_revise.

Run:
    pip install 'loopgain[examples]'
    python examples/01_code_pytest.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from loopgain import LoopGain

from _common import (
    call_claude,
    get_client,
    print_comparison,
    print_iteration,
    print_result,
    send_telemetry,
)

WORKLOAD_ID = "example-01-code-pytest"
FIXED_CAP = 5  # baseline iteration cap (also LoopGain's max_iterations)

SPEC = """\
Write a Python module that defines a single function:

    format_duration(seconds: int) -> str

It converts a non-negative integer number of seconds into a human-readable
English duration. The rules:

  - 0 seconds returns the literal string "now".
  - Negative inputs raise ValueError.
  - Units in order, dropping zero components:
      year = 365 days, day = 24 hours, hour = 60 minutes, minute = 60 seconds.
  - Oxford-comma join: [a] -> "a"; [a, b] -> "a and b";
    [a, b, c] -> "a, b and c"; [a, b, c, d] -> "a, b, c and d".
  - Each component is "<n> <unit>"; unit pluralized iff n != 1.

Return ONLY the Python source for `solution.py`. No prose, no fences.
"""

TESTS = '''\
import pytest
from solution import format_duration


@pytest.mark.parametrize("seconds,expected", [
    (0,                                "now"),
    (1,                                "1 second"),
    (62,                               "1 minute and 2 seconds"),
    (120,                              "2 minutes"),
    (3600,                             "1 hour"),
    (3662,                             "1 hour, 1 minute and 2 seconds"),
    (86400,                            "1 day"),
    (86400 + 3600,                     "1 day and 1 hour"),
    (86400 + 1,                        "1 day and 1 second"),
    (86400 * 365,                      "1 year"),
    (86400 * 365 + 86400,              "1 year and 1 day"),
    (86400 * 365 * 2 + 86400 + 1,      "2 years, 1 day and 1 second"),
    (132030240, "4 years, 68 days, 3 hours and 4 minutes"),
    (15731080,  "182 days, 1 hour, 44 minutes and 40 seconds"),
])
def test_format(seconds, expected):
    assert format_duration(seconds) == expected


def test_negative_raises():
    with pytest.raises(ValueError):
        format_duration(-1)
'''


def strip_code_fences(text: str) -> str:
    fence = re.match(r"^\s*```(?:python|py)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return fence.group(1) if fence else text


def run_pytest(workdir: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header"],
        cwd=workdir, capture_output=True, text=True, timeout=30,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+)\s+failed", out)
    failures = int(m.group(1)) if m else 0
    if proc.returncode != 0 and failures == 0:
        failures = 15  # collection/import error → worst-case signal
    return failures, out


def one_iteration(client, workdir: Path, prev_code: str, prev_failures: str):
    """Single verify-revise step. Returns (failing_count, code, pytest_output)."""
    if not prev_code:
        prompt = SPEC
    else:
        prompt = (
            f"{SPEC}\n\nYour previous attempt was:\n```python\n{prev_code}\n```"
            f"\n\npytest reported these failures:\n```\n{prev_failures[-1500:]}\n```"
            f"\n\nReturn a fully corrected `solution.py`. Code only, no prose."
        )
    code = strip_code_fences(call_claude(client, prompt))
    if not code:
        return 15, "", ""
    (workdir / "solution.py").write_text(code)
    failures, output = run_pytest(workdir)
    return failures, code, output


def baseline_run(client, workdir: Path):
    """No LoopGain: run FIXED_CAP iterations unconditionally, keep last output."""
    print("─── BASELINE: no LoopGain, fixed cap = {} ───".format(FIXED_CAP))
    code, fail_out, failures = "", "", -1
    for i in range(FIXED_CAP):
        failures, code, fail_out = one_iteration(client, workdir, code, fail_out)
        print(f"  iter {i+1:>2}  error={failures:>3}  (always runs to cap)")
    print(f"  → kept LAST output. final error={failures}\n")
    return failures, FIXED_CAP


def loopgain_run(client, workdir: Path):
    """With LoopGain: short-circuit on success (target_error=0), best-so-far."""
    print("─── WITH LOOPGAIN: target_error=0, max_iterations={} ───".format(FIXED_CAP))
    lg = LoopGain(target_error=0, max_iterations=FIXED_CAP)
    code, fail_out = "", ""
    while lg.should_continue():
        failures, code, fail_out = one_iteration(client, workdir, code, fail_out)
        state = lg.observe(failures, output=code)
        print_iteration(lg.result.iterations_used, failures, state, lg.eta)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    workdir = Path(tempfile.mkdtemp(prefix="loopgain-ex01-"))
    (workdir / "test_solution.py").write_text(TESTS)
    print(f"Workdir: {workdir}")
    print(f"Spec:    format_duration — 15 parametrized test cases.\n")

    baseline_err, baseline_iters = baseline_run(client, workdir)
    # Fresh workdir for the LoopGain run so the comparison is apples-to-apples.
    workdir = Path(tempfile.mkdtemp(prefix="loopgain-ex01-lg-"))
    (workdir / "test_solution.py").write_text(TESTS)
    lg = loopgain_run(client, workdir)

    print_comparison(baseline_iters, baseline_err, lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="verify_revise")


if __name__ == "__main__":
    main()
