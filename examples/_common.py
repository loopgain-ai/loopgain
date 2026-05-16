"""Shared helpers for LoopGain examples.

Each example script in this directory wraps a real Claude verify-revise loop
with a LoopGain monitor and POSTs aggregate telemetry on completion. The
boilerplate (reading env vars, building the Anthropic client, sending
telemetry) lives here so each example file stays small and readable.

Env vars used:
    ANTHROPIC_API_KEY              - required to call Claude
    LOOPGAIN_TELEMETRY_ENDPOINT    - canonical: https://telemetry.loopgain.ai/v1/aggregate
    LOOPGAIN_TELEMETRY_TOKEN       - bearer token; if unset, telemetry is skipped
    LOOPGAIN_EXAMPLE_MODEL         - optional override (default: claude-opus-4-7)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

DEFAULT_MODEL = os.environ.get("LOOPGAIN_EXAMPLE_MODEL", "claude-opus-4-7")
DEFAULT_ENDPOINT = "https://telemetry.loopgain.ai/v1/aggregate"


def get_client():
    """Return an anthropic.Anthropic client, or exit with a friendly error."""
    try:
        import anthropic
    except ImportError:
        sys.exit(
            "anthropic SDK is not installed. Run:\n"
            "    pip install 'loopgain[examples]'\n"
            "or:\n"
            "    pip install anthropic"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Export it and retry.")
    return anthropic.Anthropic()


def call_claude(
    client,
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    """One-shot Claude call. Returns the text of the first content block.

    Wrapped to never raise: on any API error, returns an empty string so the
    caller's error_fn can treat the iteration as worst-case and let LoopGain
    decide whether to keep going.
    """
    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = client.messages.create(**kwargs)
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
    except Exception as exc:  # noqa: BLE001 — examples should never crash mid-loop
        print(f"  [claude error: {exc!r}]")
        return ""


def send_telemetry(lg, *, workload_id: str, loop_type: str) -> None:
    """POST aggregate telemetry to the receiver. No-op if token is unset."""
    token = os.environ.get("LOOPGAIN_TELEMETRY_TOKEN")
    endpoint = os.environ.get("LOOPGAIN_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)
    if not token:
        print()
        print("Skipped telemetry POST — LOOPGAIN_TELEMETRY_TOKEN is not set.")
        return
    print()
    print(f"Posting telemetry to {endpoint} (workload_id={workload_id!r})...")
    ok = lg.send_telemetry(
        endpoint=endpoint,
        token=token,
        workload_id=workload_id,
        loop_type=loop_type,
        framework=None,
    )
    print(f"telemetry sent: {ok}")


def print_iteration(i: int, error: float, state: str, eta, preview: str = "") -> None:
    """One-line trace per iteration. Truncates preview to keep output legible."""
    eta_str = str(eta) if eta is not None else "—"
    preview = (preview or "").replace("\n", " ")
    if len(preview) > 60:
        preview = preview[:57] + "..."
    print(
        f"  iter {i:>2}  error={error:6.2f}  state={state:<14}  "
        f"eta={eta_str:<3}  {preview}"
    )


def print_result(lg) -> None:
    r = lg.result
    print()
    print(f"outcome:        {r.outcome}")
    print(f"iterations:     {r.iterations_used}")
    print(f"best_error:     {r.best_error:.2f}")
    margin = f"{r.gain_margin:.3f}" if r.gain_margin else "n/a"
    print(f"gain_margin:    {margin}")
    print(f"savings:        {r.savings_vs_fixed_cap}")


def print_comparison(baseline_iters: int, baseline_err, lg) -> None:
    """Print the headline baseline-vs-LoopGain comparison block.

    Both runs are real (each iteration is a live Claude call); the savings
    number is the measured difference, not a hypothetical.
    """
    r = lg.result
    saved = baseline_iters - r.iterations_used
    pct = (saved / baseline_iters) if baseline_iters else 0.0
    best_iter = (r.best_index + 1) if r.best_index is not None else "n/a"
    print()
    print("┌─ COMPARISON " + "─" * 50)
    print(f"│  Baseline:  {baseline_iters} iters, final error = {baseline_err}, kept LAST")
    print(
        f"│  LoopGain:  {r.iterations_used} iters, best  error = {r.best_error}, "
        f"kept best-so-far (iter {best_iter}) — state {r.outcome}"
    )
    print(f"│  Saved:     {saved} iterations ({pct:.0%}) of API spend")
    print("└" + "─" * 62)


def print_rollback_note(lg) -> None:
    """For OSCILLATING / DIVERGING demos: did LoopGain rescue an earlier output?"""
    r = lg.result
    terminal_iter = r.iterations_used - 1
    rolled_back = r.best_index is not None and r.best_index < terminal_iter
    print()
    print(f"best_index:     {r.best_index}  (terminal iter was {terminal_iter})")
    print(f"rolled_back:    {rolled_back}"
          f"{'  ← LoopGain returned an EARLIER iter than the terminal one' if rolled_back else ''}")
