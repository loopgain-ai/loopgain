"""Example 03 — Short essay with LLM-rubric critique → STALLING.

Two-LLM verify-revise loop:
  1. Claude #1 drafts (or revises) a ~120-word paragraph.
  2. Claude #2 (separate call, judge role) rubric-grades it on four criteria
     and returns a JSON object {clarity, accuracy, concision, prose}.
  3. Error = 10 - average_score. Drops toward 0 as the essay improves but
     typically plateaus once Claude has covered the basics.

target_error=None: rubric loops effectively never score 10/10/10/10, so we
don't want a target-met short-circuit. The point is to let the plateau
manifest and have LoopGain catch the STALLING band.

Expected band:  STALLING — the rubric score plateaus and the loop stops
                making progress. (LLM-as-judge scoring is mildly noisy, so an
                occasional run reads CONVERGING/DIVERGING instead.)
Loop type:      verify_revise.
"""

from __future__ import annotations

import json
import re

from loopgain import LoopGain

from _common import (
    call_claude,
    get_client,
    print_comparison,
    print_iteration,
    print_result,
    send_telemetry,
)

WORKLOAD_ID = "example-03-essay-critique"
FIXED_CAP = 6  # baseline cap; LoopGain typically catches STALLING earlier

TOPIC = (
    "Explain the 2008 global financial crisis to a curious layperson in "
    "about 120 words. Cover: what triggered it, why it spread, and the "
    "broad consequences. Plain language, no jargon."
)

JUDGE_SYSTEM = (
    "You are a strict, consistent editorial judge. Score the supplied paragraph "
    "on four criteria, each from 1 (terrible) to 10 (publishable in The "
    "Economist). Apply uniform standards across iterations: do not inflate "
    "scores for incremental edits.\n\n"
    "Return ONLY a JSON object: "
    '{"clarity": int, "accuracy": int, "concision": int, "prose": int}.'
)


def parse_scores(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    payload = fence.group(1) if fence else text
    try:
        d = json.loads(payload)
    except Exception:
        return None
    keys = ("clarity", "accuracy", "concision", "prose")
    if not isinstance(d, dict) or not all(k in d for k in keys):
        return None
    try:
        return {k: float(d[k]) for k in keys}
    except (TypeError, ValueError):
        return None


def error_from_scores(scores) -> float:
    if scores is None:
        return 10.0
    return max(0.0, 10.0 - sum(scores.values()) / 4.0)


def one_iteration(client, prev_essay: str, prev_scores):
    """Draft (or revise) → judge → return (error, essay, scores)."""
    if not prev_essay:
        draft_prompt = TOPIC
    else:
        crit = (
            ", ".join(f"{k}={int(v)}" for k, v in prev_scores.items())
            if prev_scores else "no scores parsed"
        )
        draft_prompt = (
            f"{TOPIC}\n\nYour previous draft was:\n\n{prev_essay}\n\n"
            f"The editorial judge scored it: {crit} (out of 10 each). "
            "Produce a revised version that lifts the weakest scores while "
            "keeping length near 120 words. Prose only."
        )
    essay = call_claude(client, draft_prompt, max_tokens=400)
    if not essay:
        return 10.0, "", None
    judge_reply = call_claude(
        client,
        f"Paragraph to score:\n\n{essay}",
        system=JUDGE_SYSTEM,
        max_tokens=128,
    )
    scores = parse_scores(judge_reply)
    return error_from_scores(scores), essay, scores


def baseline_run(client):
    """No LoopGain: run FIXED_CAP iterations unconditionally."""
    print(f"─── BASELINE: no LoopGain, fixed cap = {FIXED_CAP} ───")
    essay, scores, err = "", None, 10.0
    for i in range(FIXED_CAP):
        err, essay, scores = one_iteration(client, essay, scores)
        avg = 10.0 - err
        print(f"  iter {i+1:>2}  error={err:5.2f}  (avg score {avg:4.2f})")
    print(f"  → kept LAST output. final error={err:.2f}\n")
    return err, FIXED_CAP


def loopgain_run(client):
    """With LoopGain: target_error=None so STALLING (not TARGET_MET) terminates."""
    print(f"─── WITH LOOPGAIN: target_error=None, max_iterations={FIXED_CAP} ───")
    lg = LoopGain(target_error=None, max_iterations=FIXED_CAP)
    essay, scores = "", None
    while lg.should_continue():
        err, essay, scores = one_iteration(client, essay, scores)
        avg = 10.0 - err
        preview = f"avg={avg:.2f}; " + (essay.splitlines()[0] if essay else "")
        state = lg.observe(err, output=essay)
        print_iteration(lg.result.iterations_used, err, state, lg.eta, preview)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    print("Topic: 2008 financial crisis, ~120 words, layperson audience.\n")
    baseline_err, baseline_iters = baseline_run(client)
    lg = loopgain_run(client)
    print_comparison(baseline_iters, baseline_err, lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="verify_revise")


if __name__ == "__main__":
    main()
