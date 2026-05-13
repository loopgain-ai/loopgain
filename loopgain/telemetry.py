"""Anonymized telemetry emission for LoopGain.

Opt-in. Sends a single POST per loop run to a customer-configured endpoint.
Privacy: only structural statistics (state transitions, Aβ summary, gain margin,
rollback flag, library version, optional opaque workload label) are sent.
Never sends prompts, completions, error contents, or customer identity beyond
the bearer token.

The hosted endpoint at ``telemetry.loopgain.ai`` is one acceptable
destination; the receiver code is open-source so users can also self-host
to keep the data fully under their control.
"""

from __future__ import annotations

import json
import statistics
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from loopgain.core import LoopGain


# Schema version is incremented when the payload format breaks compatibility.
SCHEMA_VERSION = 1

# Library version (kept in sync with __init__.py).
LIBRARY_VERSION = "0.1.2"


def build_payload(
    lg: "LoopGain",
    workload_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> dict[str, Any]:
    """Construct the anonymized telemetry payload from a LoopGain instance.

    Args:
        lg: The LoopGain instance to summarize. Should be at terminal state,
            though mid-loop instances are also supported (outcome will be
            ``"in_progress"``).
        workload_id: Optional opaque customer-controlled string that groups
            related loops in the dashboard. Never used to identify the
            customer. Default ``None``.
        timestamp: When the loop ran. Defaults to current UTC, hour-bucketed.

    Returns:
        A JSON-serializable dict matching the v1 telemetry schema.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    # Hour-bucket to coarsen the timestamp before transmission.
    hour_bucket = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()

    result = lg.result

    # Summarize convergence profile (no individual Aβ values transmitted —
    # min / max / median are enough for the Convergence Profiles dashboard).
    profile = result.convergence_profile
    if profile:
        profile_summary = {
            "min": min(profile),
            "max": max(profile),
            "median": statistics.median(profile),
            "samples": len(profile),
        }
    else:
        profile_summary = {"min": None, "max": None, "median": None, "samples": 0}

    return {
        "schema_version": SCHEMA_VERSION,
        "library": "loopgain",
        "library_version": LIBRARY_VERSION,
        "workload_id": workload_id,
        "timestamp_hour": hour_bucket,
        "loop": {
            "outcome": result.outcome,
            "iterations_used": result.iterations_used,
            "gain_margin": result.gain_margin,
            "savings_vs_fixed_cap": result.savings_vs_fixed_cap,
            "convergence_profile_summary": profile_summary,
            "rollback_triggered": result.outcome in ("oscillating", "diverged"),
        },
        "thresholds": {
            "fast_converge": lg.thresholds.fast_converge,
            "converging": lg.thresholds.converging,
            "stalling": lg.thresholds.stalling,
            "oscillating_upper": lg.thresholds.oscillating_upper,
        },
        "smoothing_window": lg.smoothing_window,
    }


def send_payload(
    endpoint: str,
    token: str,
    payload: dict[str, Any],
    timeout: float = 2.0,
) -> bool:
    """POST a telemetry payload to the given endpoint.

    Best-effort: errors are swallowed; never raises. Returns ``True`` if
    the server returned a 2xx status, ``False`` otherwise.

    Args:
        endpoint: Telemetry receiver URL (e.g.,
            ``https://telemetry.loopgain.ai/v1/aggregate``).
        token: Bearer token issued by the receiver. Identifies the customer
            account; rotatable; not linked to any production secrets.
        payload: Dict from ``build_payload``.
        timeout: Per-request timeout in seconds. Default 2.0.

    Returns:
        ``True`` on 2xx response, ``False`` otherwise.
    """
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": f"loopgain/{LIBRARY_VERSION}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        # Best-effort: never break the user's loop because telemetry failed.
        # Catches URLError, HTTPError, TimeoutError, OSError, plus the
        # ValueError that urllib raises for malformed URLs (e.g., missing scheme),
        # plus any JSON-encoding edge case in the payload.
        return False
