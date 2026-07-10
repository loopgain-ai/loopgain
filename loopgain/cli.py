"""``loopgain`` command-line interface.

Small, dependency-free CLI whose main job is to make the opt-in *funnel*
telemetry (see :mod:`loopgain.funnel`) inspectable and controllable from the
shell — the transparency half of "opt-in, default-decline":

    loopgain telemetry --show       # what would be sent, and whether it's on
    loopgain telemetry --enable     # opt in to anonymous funnel telemetry
    loopgain telemetry --disable    # opt out
    loopgain telemetry --reset      # forget instance id + consent
    loopgain version                # print the library version
    loopgain doctor                 # send one synthetic test event end-to-end

This is intentionally separate from the product telemetry receiver: there is
no token here and nothing about a customer's loop data.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from loopgain._version import __version__
from loopgain.funnel import Funnel


def _print_status() -> None:
    # No background thread for a one-shot CLI call.
    s = Funnel(start_background=False).status()
    state = "ON" if s["enabled"] else "OFF"
    print(f"loopgain funnel telemetry: {state}  (mode: {s['mode']})")
    print(f"  decided by:    {s['consent_source']}")
    print(f"  instance id:   {s['instance_id'] or '(none yet)'}")
    print(f"  endpoint:      {s['endpoint']}")
    print(f"  config file:   {s['config_file']}")
    print(f"  first init:    {s['first_init_at'] or '(not recorded)'}")
    print(f"  first observe: {s['first_observe_at'] or '(not recorded)'}")
    print(f"  sessions:      {s['session_count']}")
    print()
    print("What would be sent (anonymous, only when ON):")
    print("  • a random instance id (not derived from your machine or identity)")
    print("  • hour-bucketed timestamps for: first init, first observe(), each session")
    print("  • library / Python / OS versions and which framework adapter was used")
    print("  • a coarse count of loop outcomes (converged / diverged / ...)")
    print("What is NEVER sent: prompts, outputs, error contents, keys, paths, IPs.")
    print()
    print("Change it:  loopgain telemetry --enable | --disable")
    print("            or set LOOPGAIN_TELEMETRY=1|0  (DO_NOT_TRACK=1 also opts out)")
    print("Details:    TELEMETRY.md")
    if not s["enabled"]:
        print()
        print("If LoopGain is useful to you, opting in is the cheapest way to support")
        print("the project — these counts are the only adoption signal it has.")


def _handle_telemetry(args: argparse.Namespace) -> int:
    if args.enable:
        Funnel(start_background=False).set_consent(True)
        print("Anonymous funnel telemetry ENABLED. Thank you — see TELEMETRY.md.")
        return 0
    if args.disable:
        Funnel(start_background=False).set_consent(False)
        print("Anonymous funnel telemetry DISABLED. Nothing will be sent.")
        return 0
    if args.reset:
        Funnel(start_background=False).reset()
        print("Funnel telemetry state reset (instance id + consent forgotten).")
        return 0
    _print_status()
    return 0


def _handle_doctor(args: "argparse.Namespace") -> int:
    """Prove the telemetry pipeline end-to-end with ONE synthetic event.

    Runs a tiny in-process verify-revise loop (no model calls, no cost),
    then sends its telemetry to the configured receiver. If this prints
    "event accepted", the user's token + endpoint + network path all work
    and the run appears in their dashboard within one refresh interval.
    The event is labeled team="doctor" so it is filterable (and excluded
    from nothing — it is a real, honest test event).
    """
    from loopgain import LoopGain
    from loopgain.telemetry import resolve_telemetry_config

    resolved = resolve_telemetry_config(args.endpoint, args.token)
    if resolved is None:
        print("loopgain doctor: no receiver configured.")
        print("  Set LOOPGAIN_TELEMETRY_ENDPOINT and LOOPGAIN_TELEMETRY_TOKEN,")
        print("  or pass --endpoint / --token.")
        print("  Free hosted token: https://loopgain.ai/#pricing (Individual tier).")
        return 2
    endpoint, _tok = resolved
    print(f"loopgain doctor · receiver: {endpoint}")

    print("  1/3 running a synthetic 3-iteration loop (no model calls)…")
    lg = LoopGain(target_error=0.1, max_iterations=5)
    for err in (1.0, 0.4, 0.05):
        if not lg.should_continue():
            break
        lg.observe(err, f"synthetic-output-{err}")
    res = lg.result
    print(f"      outcome: {res.outcome} · {res.iterations_used} iterations")

    print("  2/3 sending telemetry (one event, team='doctor')…")
    ok = lg.send_telemetry(
        endpoint=args.endpoint,
        token=args.token,
        workload_id="loopgain-doctor",
        team="doctor",
        allow_insecure=bool(args.allow_insecure),
    )
    if not ok:
        print("      ✗ send failed — check the token, endpoint, and network.")
        print("        (401 → wrong/rotated token · 404 → wrong endpoint path)")
        return 1
    print("      ✓ event accepted by the receiver")
    print("  3/3 done — open your dashboard; the run appears within one refresh:")
    print("      https://dashboard.loopgain.ai  (Recent runs → loopgain-doctor)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loopgain",
        description="LoopGain — Barkhausen stability monitor for AI agent loops.",
    )
    parser.add_argument(
        "--version", action="version", version=f"loopgain {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    p_tel = sub.add_parser(
        "telemetry",
        help="show or change opt-in anonymous funnel telemetry",
        description=(
            "Inspect and control opt-in anonymous funnel telemetry. With no "
            "flag, prints the current status (the default action)."
        ),
    )
    grp = p_tel.add_mutually_exclusive_group()
    grp.add_argument("--show", action="store_true", help="show current status (default)")
    grp.add_argument("--enable", action="store_true", help="opt in to anonymous telemetry")
    grp.add_argument("--disable", action="store_true", help="opt out of telemetry")
    grp.add_argument("--reset", action="store_true", help="forget instance id + consent")

    sub.add_parser("version", help="print the library version")

    p_doc = sub.add_parser(
        "doctor",
        help="send one synthetic test event to verify your telemetry setup",
        description=(
            "Runs a tiny in-process loop (no model calls, $0) and sends its "
            "telemetry to your receiver, proving token + endpoint + network "
            "end-to-end. Reads LOOPGAIN_TELEMETRY_ENDPOINT / _TOKEN unless "
            "--endpoint/--token are given."
        ),
    )
    p_doc.add_argument("--endpoint", default=None, help="receiver base URL or /v1/aggregate URL")
    p_doc.add_argument("--token", default=None, help="bearer token (lgk_…)")
    p_doc.add_argument(
        "--allow-insecure",
        dest="allow_insecure",
        action="store_true",
        help="permit http:// endpoints (local receivers)",
    )

    args = parser.parse_args(argv)

    if args.command == "telemetry":
        return _handle_telemetry(args)
    if args.command == "doctor":
        return _handle_doctor(args)
    if args.command == "version":
        print(__version__)
        return 0

    # No subcommand: print help and exit non-zero (nothing to do).
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
