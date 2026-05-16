"""Example 06 — Compounding-error refinement → DIVERGING + rollback.

Each iteration asks Claude to make the passage shorter, with no instruction
to preserve facts. Each pass strips information; error climbs monotonically.

Error signal: a fixed list of factual atoms (specific dates, dollar amounts,
named people, drug code, trial phase). We substring-match each atom against
the current draft; error = number of atoms missing.

This is the **other headline rollback demo**. The naive baseline keeps the
last iteration (worst output). LoopGain detects Aβ > 1.05, aborts, and
returns the best earlier iteration from the rollback buffer.

target_error=None: error 0 is reachable in principle (a perfectly
preserved short rewrite) but the "make it shorter" prompt actively works
against that — we don't expect a target-met short-circuit and we want the
DIVERGING band to fire when the loop starts losing facts.

Expected band:  DIVERGING with best_index < iterations_used - 1.
Loop type:      refinement.
"""

from __future__ import annotations

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

WORKLOAD_ID = "example-06-diverges"
FIXED_CAP = 8

PASSAGE = (
    "On April 7, 2024, biotech startup NovaGen Therapeutics announced it had "
    "raised $185 million in Series C funding led by Andreessen Horowitz. The "
    "round was joined by existing investor Founders Fund and brought the "
    "company's total funding to $312 million. CEO Dr. Elena Martinez stated "
    "that the capital would accelerate development of their lead drug "
    "candidate, NVG-401, currently in Phase 2 trials for treating "
    "glioblastoma. The company plans to expand its 87-person team to 150 by "
    "year-end and open a second research facility in Cambridge, Massachusetts."
)

FACTS = [
    "April 7, 2024",
    "$185 million",
    "NovaGen",
    "Andreessen Horowitz",
    "$312 million",
    "Elena Martinez",
    "NVG-401",
    "Phase 2",
]


def count_missing(text: str):
    lower = (text or "").lower()
    missing = [f for f in FACTS if f.lower() not in lower]
    return len(missing), missing


def one_iteration(client, prev_text: str, iteration: int):
    if iteration == 1:
        prompt = (
            f"Rewrite this passage to be 40 words long. Return only the "
            f"rewritten passage, no preamble.\n\n{prev_text}"
        )
    else:
        prompt = (
            f"Make this even shorter. Return only the rewritten passage, "
            f"no preamble.\n\n{prev_text}"
        )
    revised = call_claude(client, prompt, max_tokens=400)
    if not revised:
        return len(FACTS), prev_text, ["[no text]"]
    n, missing = count_missing(revised)
    return n, revised, missing


def baseline_run(client):
    print(f"─── BASELINE: no LoopGain, fixed cap = {FIXED_CAP} ───")
    current = PASSAGE
    err = 0
    for i in range(FIXED_CAP):
        err, current, missing = one_iteration(client, current, i + 1)
        wc = len(current.split())
        print(f"  iter {i+1:>2}  error={err}  ({wc}w, missing {err}/{len(FACTS)})")
    print(f"  → kept LAST output (terminal iter, error={err}).\n")
    return err, FIXED_CAP


def loopgain_run(client):
    print(f"─── WITH LOOPGAIN: target_error=None, max_iterations={FIXED_CAP} ───")
    lg = LoopGain(target_error=None, max_iterations=FIXED_CAP)
    current = PASSAGE
    iteration = 0
    while lg.should_continue():
        iteration += 1
        err, current, missing = one_iteration(client, current, iteration)
        wc = len(current.split())
        preview = f"{wc}w; missing={err}/{len(FACTS)}: {missing[:2]}"
        state = lg.observe(err, output=current)
        print_iteration(lg.result.iterations_used, err, state, lg.eta, preview)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    print(f"Spec: shorten a press blurb (8 fact atoms tracked).\n"
          "(Diverges by construction — expecting DIVERGING + rollback.)\n")
    baseline_err, baseline_iters = baseline_run(client)
    lg = loopgain_run(client)
    print_comparison(baseline_iters, baseline_err, lg)
    print_rollback_note(lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="refinement")


if __name__ == "__main__":
    main()
