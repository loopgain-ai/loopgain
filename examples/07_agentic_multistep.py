"""Example 07 — Agentic multi-step reasoning toward a goal.

The "LangChain-style agentic loop" people typically picture when they hear
"AI agent loop." Claude is given a goal with multiple structured
constraints and reasons toward it across iterations:

  1. Claude proposes a meal plan as a JSON object.
  2. A deterministic verifier checks N constraints against a fixture
     pantry (ingredient set, price-per-100g, required servings, allowed
     techniques, step count, etc.).
  3. Failed constraints are fed back as a structured critique.
  4. Error = number of constraints unmet. 0 fires TARGET_MET.

Unlike 01-04 (single verifier on a single output dimension), this loop
involves multi-dimensional constraint reasoning — the kind of structured
goal-pursuit that distinguishes "agentic" loops from straight verify-revise.

Expected band:  CONVERGING → TARGET_MET in 1-3 iterations.
Loop type:      multi_step_reasoning.
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

WORKLOAD_ID = "example-07-agentic-multistep"
FIXED_CAP = 5

# Fixture "world state" the agent has to reason against.
PANTRY = {
    "chicken_thigh":   {"price_per_100g": 1.20},
    "onion":           {"price_per_100g": 0.30},
    "garlic":          {"price_per_100g": 0.80},
    "rice":            {"price_per_100g": 0.40},
    "olive_oil":       {"price_per_100g": 1.50},
    "tomato":          {"price_per_100g": 0.60},
    "lemon":           {"price_per_100g": 0.50},
    "parsley":         {"price_per_100g": 2.00},
}
ALLOWED_TECHNIQUES = {"searing", "braising", "roasting", "poaching"}
GOAL = {
    "servings": 4,
    "max_total_cost_usd": 12.00,
    "min_ingredients": 4,
    "max_ingredients": 6,
    "min_steps": 5,
    "max_steps": 8,
}

SYSTEM = (
    "You produce structured meal plans. Return ONLY a single JSON object "
    "matching this shape exactly:\n"
    '{"ingredients": [{"name": "<pantry_id>", "quantity_g": <int>}], '
    '"technique": "<technique>", "servings": <int>, "steps": ["<step>", ...]}\n'
    "No prose, no fences."
)


def render_prompt(prev_plan: str, prev_errors: list[str]) -> str:
    pantry_lines = "\n".join(
        f"  - {k}: ${v['price_per_100g']:.2f} / 100g" for k, v in PANTRY.items()
    )
    base = (
        f"Goal: a dinner for {GOAL['servings']} people, total ingredient "
        f"cost under ${GOAL['max_total_cost_usd']:.2f}, using "
        f"{GOAL['min_ingredients']}-{GOAL['max_ingredients']} pantry "
        f"ingredients (no substitutions or additions outside the pantry), "
        f"applying ONE technique from {sorted(ALLOWED_TECHNIQUES)}, written "
        f"as {GOAL['min_steps']}-{GOAL['max_steps']} numbered steps.\n\n"
        f"Pantry (price-per-100g):\n{pantry_lines}\n"
    )
    if not prev_plan:
        return base + "\nProduce the plan."
    return (
        base + f"\nYour previous plan was:\n{prev_plan}\n\n"
        f"Verifier reported these unmet constraints:\n- "
        + "\n- ".join(prev_errors)
        + "\n\nProduce a corrected plan. Same JSON shape."
    )


def parse_plan(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    payload = fence.group(1) if fence else text
    try:
        return json.loads(payload)
    except Exception:
        return None


def verify(plan) -> list[str]:
    """Return the list of unmet constraints. Empty list = goal achieved."""
    errs: list[str] = []
    if not isinstance(plan, dict):
        return ["root: expected JSON object"]
    ings = plan.get("ingredients")
    if not isinstance(ings, list):
        errs.append("ingredients: expected array")
        ings = []
    bad_names = [i.get("name") for i in ings if i.get("name") not in PANTRY]
    if bad_names:
        errs.append(f"ingredients: not in pantry: {bad_names}")
    n_ings = len(ings)
    if not (GOAL["min_ingredients"] <= n_ings <= GOAL["max_ingredients"]):
        errs.append(
            f"ingredients: count {n_ings} outside "
            f"[{GOAL['min_ingredients']}, {GOAL['max_ingredients']}]"
        )
    cost = 0.0
    for i in ings:
        info = PANTRY.get(i.get("name"))
        q = i.get("quantity_g")
        if info and isinstance(q, (int, float)) and q > 0:
            cost += info["price_per_100g"] * q / 100.0
    if cost > GOAL["max_total_cost_usd"]:
        errs.append(f"cost: ${cost:.2f} exceeds cap ${GOAL['max_total_cost_usd']:.2f}")
    tech = plan.get("technique")
    if tech not in ALLOWED_TECHNIQUES:
        errs.append(f"technique: {tech!r} not in {sorted(ALLOWED_TECHNIQUES)}")
    if plan.get("servings") != GOAL["servings"]:
        errs.append(f"servings: expected {GOAL['servings']}, got {plan.get('servings')}")
    steps = plan.get("steps")
    if not isinstance(steps, list):
        errs.append("steps: expected array")
    elif not (GOAL["min_steps"] <= len(steps) <= GOAL["max_steps"]):
        errs.append(
            f"steps: count {len(steps)} outside "
            f"[{GOAL['min_steps']}, {GOAL['max_steps']}]"
        )
    return errs


def one_iteration(client, prev_raw: str, prev_errs: list[str]):
    raw = call_claude(client, render_prompt(prev_raw, prev_errs), system=SYSTEM)
    plan = parse_plan(raw)
    errs = ["root: response was not parseable JSON"] if plan is None else verify(plan)
    return len(errs), raw, errs


def baseline_run(client):
    print(f"─── BASELINE: no LoopGain, fixed cap = {FIXED_CAP} ───")
    raw, errs, n = "", [], 0
    for i in range(FIXED_CAP):
        n, raw, errs = one_iteration(client, raw, errs)
        print(f"  iter {i+1:>2}  unmet={n:>2}  (always runs to cap)")
    print(f"  → kept LAST plan. final unmet={n}\n")
    return n, FIXED_CAP


def loopgain_run(client):
    print(f"─── WITH LOOPGAIN: target_error=0, max_iterations={FIXED_CAP} ───")
    lg = LoopGain(target_error=0, max_iterations=FIXED_CAP)
    raw, errs = "", []
    while lg.should_continue():
        n, raw, errs = one_iteration(client, raw, errs)
        preview = f"{n} unmet; " + (errs[0] if errs else "goal achieved")
        state = lg.observe(n, output=raw)
        print_iteration(lg.result.iterations_used, n, state, lg.eta, preview)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    print("Goal: a structured meal plan satisfying ~7 multi-dimensional constraints.\n")
    baseline_err, baseline_iters = baseline_run(client)
    lg = loopgain_run(client)
    print_comparison(baseline_iters, baseline_err, lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="multi_step_reasoning")


if __name__ == "__main__":
    main()
