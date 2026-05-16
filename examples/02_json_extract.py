"""Example 02 — Structured JSON extraction with schema validation.

Real verify-revise loop. Each iteration:
  1. Claude reads an unstructured blurb describing a conference.
  2. Claude returns a JSON document.
  3. A small custom validator (no `jsonschema` dep) checks types, required
     fields, ISO date pattern, URL pattern.
  4. Error = number of validation errors. 0 fires TARGET_MET.

This example runs the SAME loop twice so you can compare:

  BASELINE — fixed cap of iterations regardless of success.
  LOOPGAIN — target_error=0 short-circuits the moment validation passes.

Expected band:  CONVERGING (or FAST_CONVERGE → TARGET_MET if Opus one-shots).
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

WORKLOAD_ID = "example-02-json-extract"
FIXED_CAP = 5

BLURB = """\
On March 15, 2026, the third annual MidwestPyCon conference will be held at
the Loring Park Convention Center, located at 1801 Bryant Avenue South in
Minneapolis. The event runs from 9 AM to 5 PM and features talks by Sarah
Chen on async generators, Marcus Webb on type system internals, and Priya
Rao on PyPI security. General admission tickets are $249, with a 20%
early-bird discount available until February 28. Registration is open at
https://midwestpycon.example.com/2026/register, and capacity is limited to
400 attendees.
"""

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HTTPS_URL = re.compile(r"^https?://[^\s]+$")

SYSTEM = (
    "You extract structured data. Return ONLY a single JSON object. No prose, "
    "no code fences."
)


def validate(doc) -> list[str]:
    """Return a list of human-readable validation errors. Empty = valid."""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["root: expected object"]

    def need(key, typ, path="$"):
        if key not in doc:
            errs.append(f"{path}.{key}: missing required field")
            return False
        if not isinstance(doc[key], typ):
            errs.append(f"{path}.{key}: expected {typ.__name__}, got {type(doc[key]).__name__}")
            return False
        return True

    need("name", str)
    if need("date", str) and not ISO_DATE.match(doc["date"]):
        errs.append("$.date: must match ISO 8601 YYYY-MM-DD")

    if need("venue", dict):
        v = doc["venue"]
        for k in ("name", "street_address", "city"):
            if k not in v:
                errs.append(f"$.venue.{k}: missing required field")
            elif not isinstance(v[k], str):
                errs.append(f"$.venue.{k}: expected str")

    if need("speakers", list):
        if len(doc["speakers"]) < 1:
            errs.append("$.speakers: must have at least one entry")
        for i, s in enumerate(doc["speakers"]):
            if not isinstance(s, dict):
                errs.append(f"$.speakers[{i}]: expected object")
                continue
            for k in ("name", "topic"):
                if k not in s or not isinstance(s[k], str):
                    errs.append(f"$.speakers[{i}].{k}: missing or not a string")

    if need("ticket_price_usd", (int, float)):
        if isinstance(doc["ticket_price_usd"], bool):
            errs.append("$.ticket_price_usd: must be number, not bool")
        elif doc["ticket_price_usd"] <= 0:
            errs.append("$.ticket_price_usd: must be positive")

    if need("max_attendees", int) and not isinstance(doc["max_attendees"], bool):
        if doc["max_attendees"] <= 0:
            errs.append("$.max_attendees: must be positive integer")

    if need("registration_url", str) and not HTTPS_URL.match(doc["registration_url"]):
        errs.append("$.registration_url: must match ^https?://...")
    return errs


def parse_json(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    payload = fence.group(1) if fence else text
    try:
        return json.loads(payload)
    except Exception:
        return None


def one_iteration(client, prev_json: str, prev_errors: list[str]):
    """Single extract → validate step. Returns (err_count, raw_json, errors)."""
    if not prev_json:
        prompt = (
            f"Extract the structured details from this blurb into JSON.\n\n"
            f"BLURB:\n{BLURB}\n\n"
            "Required fields: name (str), date (str, YYYY-MM-DD), venue (object "
            "with name, street_address, city), speakers (array of {name, topic}), "
            "ticket_price_usd (number, no currency symbol), max_attendees (integer), "
            "registration_url (str starting with http:// or https://)."
        )
    else:
        prompt = (
            f"Your previous JSON was:\n{prev_json}\n\n"
            f"Validator reported these errors:\n- " + "\n- ".join(prev_errors)
            + "\n\nReturn a corrected JSON document. Same fields. JSON only."
        )
    raw = call_claude(client, prompt, system=SYSTEM)
    doc = parse_json(raw)
    errs = ["root: response was not parseable JSON"] if doc is None else validate(doc)
    return len(errs), raw, errs


def baseline_run(client):
    """No LoopGain: run FIXED_CAP iterations unconditionally, keep last output."""
    print(f"─── BASELINE: no LoopGain, fixed cap = {FIXED_CAP} ───")
    raw, errs, n = "", [], 0
    for i in range(FIXED_CAP):
        n, raw, errs = one_iteration(client, raw, errs)
        print(f"  iter {i+1:>2}  error={n:>3}  (always runs to cap)")
    print(f"  → kept LAST output. final error={n}\n")
    return n, FIXED_CAP


def loopgain_run(client):
    """With LoopGain: target_error=0 short-circuits on valid JSON."""
    print(f"─── WITH LOOPGAIN: target_error=0, max_iterations={FIXED_CAP} ───")
    lg = LoopGain(target_error=0, max_iterations=FIXED_CAP)
    raw, errs = "", []
    while lg.should_continue():
        n, raw, errs = one_iteration(client, raw, errs)
        preview = f"{n} err; " + (errs[0] if errs else "valid")
        state = lg.observe(n, output=raw)
        print_iteration(lg.result.iterations_used, n, state, lg.eta, preview)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    print("Spec: extract a conference blurb into a 7-field JSON document.\n")
    baseline_err, baseline_iters = baseline_run(client)
    lg = loopgain_run(client)
    print_comparison(baseline_iters, baseline_err, lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="verify_revise")


if __name__ == "__main__":
    main()
