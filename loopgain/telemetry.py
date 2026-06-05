"""Anonymized telemetry emission for LoopGain.

Opt-in. Sends a single POST per loop run to a customer-configured endpoint.
Privacy: only structural statistics — Aβ values, error magnitudes, state
transitions, gain margin, rollback flag, library version, optional opaque
workload/classification labels. Never sends prompts, completions, error
contents (the textual content of failures), customer identity beyond the
bearer token, or best-so-far outputs.

Per-iteration trajectories (the smoothed Aβ series and error-magnitude
series) are included by default since they drive the Loop Detail scrubber
in the dashboard. They are purely numerical and contain no customer
content. Pass ``include_per_iteration=False`` to ``build_payload`` /
``LoopGain.send_telemetry`` to send only the aggregate summary.

The hosted endpoint at ``telemetry.loopgain.ai`` is one acceptable
destination; the receiver code is open-source so users can also self-host
to keep the data fully under their control.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse


def _safe_float(x: Any) -> Any:
    """Coerce inf / -inf / NaN to None so the payload stays strict JSON.

    Standard JSON (RFC 8259) forbids Infinity and NaN literals. Python's
    json.dumps emits them by default, and strict parsers — including the
    Cloudflare-side receiver — reject the payload. gain_margin in particular
    is 1/max(Aβ_smooth) and goes to +inf whenever the smoothed gain is zero
    (e.g. a constant-error trajectory). Aβ values themselves can go to inf
    if a previous error is exactly zero. Collapsing to None keeps the
    dashboard's "no data" semantics intact instead of dropping the whole
    payload.
    """
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow HTTP redirects.

    ``urllib`` follows 3xx responses by default and does NOT strip the
    Authorization header on cross-origin redirects. If the configured
    telemetry endpoint were compromised, it could 302 to ``attacker.com``
    and harvest the bearer token. We treat any 3xx as a failed delivery —
    the caller's loop is not affected (``send_payload`` swallows it).
    """

    def http_error_301(self, req, fp, code, msg, headers):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _open_request(req: urllib.request.Request, timeout: float) -> Any:
    """Single seam for the outbound HTTP call.

    Production uses ``_NO_REDIRECT_OPENER`` so the bearer can never be
    leaked via a 30x. Tests monkey-patch this function (not
    ``urllib.request.urlopen``) when they need to inspect the outgoing
    request without doing real network I/O.
    """
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout)

if TYPE_CHECKING:
    from loopgain.core import LoopGain


# Schema version is incremented when the payload format breaks compatibility.
# v2 (2026-05-13) adds first_eta_prediction + first_eta_at_iteration for the
# ETA Accuracy dashboard panel. v3 (2026-05-14) adds the optional
# per_iteration block (capped trajectories) and the framework/loop_type/team
# classification fields. Receiver remains backward-compatible: v1/v2 payloads
# are still accepted (new fields default to None / NULL).
SCHEMA_VERSION = 3


# Library version sourced from loopgain._version so there's exactly one
# string to bump per release. _version.py has no project imports, so this
# is safe to import at module load.
from loopgain._version import __version__ as LIBRARY_VERSION

# Cap on per-iteration trajectory length sent to telemetry. Loops longer than
# this are truncated to the first PER_ITERATION_CAP entries with a
# ``truncated: true`` flag in the payload. 256 is well above the typical
# 5-15 iterations of a converging loop and bounds the payload size at
# ~6 KB even for very long traces.
PER_ITERATION_CAP = 256


def build_payload(
    lg: "LoopGain",
    workload_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    framework: Optional[str] = None,
    loop_type: Optional[str] = None,
    team: Optional[str] = None,
    include_per_iteration: bool = True,
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
        framework: Optional classification label naming the agent framework
            (``"langgraph"``, ``"crewai"``, ``"autogen"``, etc.). Adapters
            auto-stamp this; raw API users may pass it manually.
        loop_type: Optional classification label naming the loop pattern
            (``"verify_revise"``, ``"rag_refine"``, ``"tool_use_retry"``,
            etc.). Free-form; used for filtering in the dashboard.
        team: Optional opaque label grouping by team or environment
            (``"prod"``, ``"team-search"``, etc.). Used for filtering only.
        include_per_iteration: If ``True`` (default), the payload includes
            the smoothed Aβ trajectory and the error-magnitude trajectory
            (capped at ``PER_ITERATION_CAP`` entries with a ``truncated``
            flag). Set ``False`` to send only aggregate summary stats.

    Returns:
        A JSON-serializable dict matching the v3 telemetry schema.
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
            "min": _safe_float(min(profile)),
            "max": _safe_float(max(profile)),
            "median": _safe_float(statistics.median(profile)),
            "samples": len(profile),
        }
    else:
        profile_summary = {"min": None, "max": None, "median": None, "samples": 0}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "library": "loopgain",
        "library_version": LIBRARY_VERSION,
        "workload_id": workload_id,
        "timestamp_hour": hour_bucket,
        # v3 classification fields. All optional; NULL on the receiver when
        # not provided. Used to drive dashboard filters across panels.
        "framework": framework,
        "loop_type": loop_type,
        "team": team,
        "loop": {
            "outcome": result.outcome,
            "iterations_used": result.iterations_used,
            "gain_margin": _safe_float(result.gain_margin),
            "savings_vs_fixed_cap": result.savings_vs_fixed_cap,
            "convergence_profile_summary": profile_summary,
            "rollback_triggered": result.outcome in ("oscillating", "diverged"),
            # Index (0-based) of the lowest-error iteration. Lets the receiver
            # derive iterations-to-best (best_index+1) and iterations-past-best
            # (iterations_used-1-best_index) — the "Iteration Waste" view.
            # Privacy-safe: an integer position, no output/error content.
            "best_index": result.best_index,
            # v2: first computable eta snapshot, for ETA calibration dashboard.
            # Predicted total iterations = first_eta_at_iteration +
            # first_eta_prediction; compare to iterations_used to compute the
            # calibration error. Both are None when no prediction was made
            # (target_error=0, loop never looked convergent, etc.).
            "first_eta_prediction": result.first_eta_prediction,
            "first_eta_at_iteration": result.first_eta_at_iteration,
        },
        "thresholds": {
            "fast_converge": lg.thresholds.fast_converge,
            "converging": lg.thresholds.converging,
            "stalling": lg.thresholds.stalling,
            "oscillating_upper": lg.thresholds.oscillating_upper,
        },
        "smoothing_window": lg.smoothing_window,
    }

    if include_per_iteration:
        # v3: per-iteration trajectories drive the Loop Detail scrubber.
        # Cap to bound the payload (and therefore D1 row size); ~6 KB at the
        # cap. error_history length == iterations_used; convergence_profile
        # is one shorter (no Aβ for the first observation).
        errors = result.error_history
        ab = result.convergence_profile
        truncated = len(errors) > PER_ITERATION_CAP or len(ab) > PER_ITERATION_CAP
        payload["per_iteration"] = {
            "error_history": [_safe_float(e) for e in errors[:PER_ITERATION_CAP]],
            "convergence_profile": [_safe_float(a) for a in ab[:PER_ITERATION_CAP]],
            "truncated": truncated,
            "cap": PER_ITERATION_CAP,
        }

    return payload


def send_payload(
    endpoint: str,
    token: str,
    payload: dict[str, Any],
    timeout: float = 2.0,
    allow_insecure: bool = False,
) -> bool:
    """POST a telemetry payload to the given endpoint.

    Best-effort: errors are swallowed; never raises. Returns ``True`` if
    the server returned a 2xx status, ``False`` otherwise.

    Args:
        endpoint: Telemetry receiver URL (e.g.,
            ``https://telemetry.loopgain.ai/v1/aggregate``). Must use
            ``https://``. ``http://`` is rejected unless ``allow_insecure``
            is ``True``; all other schemes (``file://``, ``javascript:``,
            ``ftp://``, etc.) are always rejected to keep the bearer token
            from being smuggled out over an unintended channel.
        token: Bearer token issued by the receiver. Identifies the customer
            account; rotatable; not linked to any production secrets.
        payload: Dict from ``build_payload``.
        timeout: Per-request timeout in seconds. Default 2.0.
        allow_insecure: If ``True``, permit ``http://`` endpoints. Intended
            for local development against a self-hosted receiver on
            ``http://localhost``. Default ``False``.

    Returns:
        ``True`` on 2xx response, ``False`` otherwise.
    """
    # Refuse to attach the bearer token to anything but http(s); silently
    # best-effort so a misconfigured endpoint can't break the user's loop.
    try:
        scheme = urlparse(endpoint).scheme.lower()
    except Exception:
        return False
    if scheme == "https":
        pass
    elif scheme == "http" and allow_insecure:
        pass
    else:
        return False

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
        # Use the no-redirect seam so a malicious or misconfigured
        # endpoint can't 302 the bearer token to a different host.
        with _open_request(req, timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        # Best-effort: never break the user's loop because telemetry failed.
        # Catches URLError, HTTPError, TimeoutError, OSError, plus the
        # ValueError that urllib raises for malformed URLs (e.g., missing scheme),
        # plus any JSON-encoding edge case in the payload.
        return False
