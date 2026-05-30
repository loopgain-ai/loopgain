"""Opt-in, anonymous *funnel* telemetry for the LoopGain open-source library.

This is **not** the product telemetry receiver in :mod:`loopgain.telemetry`.
Keep the two cleanly separated in your head and in the code:

- :mod:`loopgain.telemetry` (``LoopGain.send_telemetry``) ships a *customer's
  own* loop data — Aβ trajectories, error magnitudes, outcomes — to a receiver
  *they* configure with *their* bearer token, so it shows up in *their*
  dashboard. It is per-loop, explicitly called, and carries an auth token.

- This module (``loopgain.funnel``) measures the *maintainer's* adoption
  funnel across the whole OSS userbase: install → first ``observe()`` →
  recurring use. It is anonymous, has no auth token, sends a handful of
  events per install (not per loop), and exists only so the project can tell
  whether anyone is actually using it.

**Posture: opt-in, default-decline.** Nothing leaves the machine unless the
user explicitly opts in (``LOOPGAIN_TELEMETRY=1`` or ``loopgain telemetry
--enable``). Until a choice is made we are "undecided": we send nothing, but
write a small local state file and print a one-time notice explaining how to
opt in. An explicit decline (``LOOPGAIN_TELEMETRY=0``, ``DO_NOT_TRACK=1``, or
``loopgain telemetry --disable``) writes nothing and shows nothing.

**What is collected** (only when enabled): a locally-generated random instance
id (a fresh ``uuid4`` — *not* derived from any hardware, user, or network
identifier), hour-bucketed event timestamps, the library/Python/OS versions,
which framework adapter was used, and a coarse count of loop outcomes. **What
is never collected:** prompts, outputs, error contents, API keys, file paths,
IP addresses, or anything that could identify a user or their data.

Everything here is best-effort and fail-silent: a bug or network failure in
funnel telemetry must never raise into — let alone break — a user's loop.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import sys
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from loopgain._version import __version__ as LIBRARY_VERSION

# Bumped only on a breaking change to the funnel event format. Independent of
# the product-receiver SCHEMA_VERSION in loopgain.telemetry.
FUNNEL_SCHEMA_VERSION = 1

# Default funnel receiver. Distinct path from the product receiver's
# /v1/aggregate so the two never share an ingestion route. Overridable for
# self-hosting / testing via LOOPGAIN_TELEMETRY_FUNNEL_ENDPOINT.
DEFAULT_FUNNEL_ENDPOINT = "https://telemetry.loopgain.ai/v1/funnel"

# Truthy / falsy spellings accepted for LOOPGAIN_TELEMETRY and DO_NOT_TRACK.
_TRUE = frozenset({"1", "true", "yes", "on", "enable", "enabled"})
_FALSE = frozenset({"0", "false", "no", "off", "disable", "disabled", ""})

# Map terminal LoopGain state names to the coarse outcome buckets we count.
# Anything unrecognized is bucketed as "other" so the distribution stays small
# and stable even if new states are added upstream.
_STATE_TO_OUTCOME = {
    "TARGET_MET": "converged",
    "OSCILLATING": "oscillating",
    "DIVERGING": "diverged",
    "MAX_ITERATIONS": "max_iterations",
    "STALLING": "stalled",
}

# Modes resolved from environment + persisted consent.
_DISABLED = "disabled"
_UNDECIDED = "undecided"
_ENABLED = "enabled"

_NOTICE = (
    "loopgain: anonymous usage telemetry is OFF — nothing is sent unless you opt in.\n"
    "  Help improve loopgain by sharing anonymous install/usage counts (no prompts,\n"
    "  outputs, paths, or identities — ever):\n"
    "      loopgain telemetry --enable      (or set LOOPGAIN_TELEMETRY=1)\n"
    "  See exactly what would be sent:  loopgain telemetry --show  (or TELEMETRY.md)\n"
    "  This notice is shown once.\n"
)


def _now_hour(clock: Callable[[], datetime]) -> str:
    """Current UTC time, bucketed to the hour, as an ISO-8601 string.

    Hour-bucketing coarsens timestamps before transmission so an event can't
    be used as a high-resolution activity fingerprint.
    """
    return (
        clock()
        .astimezone(timezone.utc)
        .replace(minute=0, second=0, microsecond=0)
        .isoformat()
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects (defense-in-depth, mirrors telemetry.py)."""

    def http_error_301(self, req, fp, code, msg, headers):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _open_request(req: urllib.request.Request, timeout: float) -> Any:
    """Single seam for the outbound HTTP call (tests monkeypatch this)."""
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout)


def _default_sender(endpoint: str, batch: dict[str, Any], timeout: float) -> bool:
    """POST a batch of funnel events. https-only, no auth, fully fail-silent.

    Returns ``True`` on a 2xx response, ``False`` on anything else. Never
    raises — the caller relies on that.
    """
    try:
        if urlparse(endpoint).scheme.lower() != "https":
            return False
    except Exception:
        return False
    try:
        body = json.dumps(batch).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"loopgain/{LIBRARY_VERSION} (funnel)",
            },
        )
        with _open_request(req, timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


class Funnel:
    """Manages opt-in funnel telemetry for a single process.

    A module-level singleton (see :func:`_instance`) is used in production and
    driven by the hooks at the bottom of this module. The class is
    self-contained and dependency-injectable (``config_dir``, ``sender``,
    ``clock``, ``start_background``) so tests can exercise it in isolation
    without touching the real config directory, network, or wall clock.
    """

    #: How often the background thread flushes queued events (seconds). Long
    #: enough to batch; short enough that a long-lived process reports
    #: first_init / first_observe without waiting for interpreter exit.
    FLUSH_INTERVAL = 20.0

    #: Per-request timeout for the outbound POST (seconds).
    SEND_TIMEOUT = 2.0

    def __init__(
        self,
        *,
        config_dir: Optional[str] = None,
        sender: Optional[Callable[[str, dict[str, Any], float], bool]] = None,
        endpoint: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
        start_background: bool = True,
    ) -> None:
        self._config_dir = config_dir or self._resolve_config_dir()
        self._sender = sender or _default_sender
        self._endpoint = (
            endpoint
            or os.environ.get("LOOPGAIN_TELEMETRY_FUNNEL_ENDPOINT")
            or DEFAULT_FUNNEL_ENDPOINT
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._start_background = start_background

        self._lock = threading.RLock()
        self._queue: list[dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._atexit_registered = False

        self._mode: Optional[str] = None  # resolved lazily on first activity
        self._state: dict[str, Any] = {}
        self._loaded = False
        self._session_started = False
        self._adapter: Optional[str] = None
        self._outcomes: dict[str, int] = {}

    # ----- Paths -----

    @staticmethod
    def _resolve_config_dir() -> str:
        explicit = os.environ.get("LOOPGAIN_CONFIG_DIR")
        if explicit:
            return explicit
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".config")
        return os.path.join(base, "loopgain")

    @property
    def _state_path(self) -> str:
        return os.path.join(self._config_dir, "funnel.json")

    # ----- State file I/O (all best-effort) -----

    def _read_state_file(self) -> dict[str, Any]:
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_state_file(self) -> None:
        try:
            os.makedirs(self._config_dir, exist_ok=True)
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, sort_keys=True)
            os.replace(tmp, self._state_path)
        except Exception:
            # A read-only or missing home directory must not break anything.
            pass

    # ----- Consent resolution -----

    def _env_consent(self) -> Optional[bool]:
        """Resolve consent from the environment alone (no file access).

        Returns ``True`` (opt-in), ``False`` (opt-out), or ``None`` (the env
        says nothing). ``DO_NOT_TRACK`` is honored as a hard opt-out and takes
        precedence over ``LOOPGAIN_TELEMETRY``.
        """
        dnt = os.environ.get("DO_NOT_TRACK")
        if dnt is not None and dnt.strip().lower() in _TRUE:
            return False
        raw = os.environ.get("LOOPGAIN_TELEMETRY")
        if raw is None:
            return None
        val = raw.strip().lower()
        if val in _TRUE:
            return True
        if val in _FALSE:
            return False
        return None  # unrecognized spelling → treat as unset

    @staticmethod
    def _is_ci() -> bool:
        # Common CI markers. In CI we stay silent and decline by default
        # (no notice, no send) — a build log is the wrong place to nag.
        return any(
            os.environ.get(k) not in (None, "", "0", "false", "False")
            for k in ("CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI")
        )

    def _resolve_mode(self) -> str:
        env = self._env_consent()
        if env is True:
            return _ENABLED
        if env is False:
            return _DISABLED
        # Env is silent → consult persisted consent (read without creating).
        persisted = self._read_state_file().get("consent")
        if persisted == "granted":
            return _ENABLED
        if persisted == "denied":
            return _DISABLED
        if self._is_ci():
            return _DISABLED
        return _UNDECIDED

    # ----- Lazy load / one-time setup -----

    def _ensure_loaded(self) -> str:
        """Resolve mode and prepare state exactly once per process.

        Returns the resolved mode. In ``disabled`` mode nothing is read,
        written, or shown. In ``undecided`` mode we ensure a state file with
        an instance id exists and show the one-time notice. In ``enabled``
        mode we additionally bump the session counter and arm the flush
        machinery.
        """
        if self._loaded:
            return self._mode  # type: ignore[return-value]
        with self._lock:
            if self._loaded:
                return self._mode  # type: ignore[return-value]
            mode = self._resolve_mode()
            self._mode = mode

            if mode == _DISABLED:
                self._loaded = True
                return mode

            # undecided or enabled: materialize state + instance id.
            self._state = self._read_state_file()
            if not self._state.get("instance_id"):
                self._state["instance_id"] = uuid.uuid4().hex
                self._state.setdefault("schema", FUNNEL_SCHEMA_VERSION)
            # Persist consent marker so the CLI / future runs are consistent.
            if mode == _ENABLED and self._state.get("consent") != "granted":
                self._state["consent"] = "granted"

            if mode == _UNDECIDED and not self._state.get("notice_shown"):
                self._maybe_show_notice()
                self._state["notice_shown"] = True

            if mode == _ENABLED:
                self._state["session_count"] = int(self._state.get("session_count", 0)) + 1
                self._session_started = True

            self._write_state_file()
            self._loaded = True
            return mode

    def _maybe_show_notice(self) -> None:
        """Print the one-time opt-in notice, TTY-gated and fail-silent."""
        try:
            if sys.stderr is not None and sys.stderr.isatty():
                sys.stderr.write(_NOTICE)
        except Exception:
            pass

    # ----- Event construction -----

    def _base_event(self, event: str) -> dict[str, Any]:
        return {
            "event": event,
            "instance_id": self._state.get("instance_id"),
            "ts_hour": _now_hour(self._clock),
            "library_version": LIBRARY_VERSION,
            "python": "%d.%d" % (sys.version_info[0], sys.version_info[1]),
            "os": platform.system() or "unknown",
        }

    def _enqueue(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._queue.append(event)
        self._ensure_thread()

    # ----- Flush machinery -----

    def _ensure_thread(self) -> None:
        if not self._start_background:
            return
        with self._lock:
            if not self._atexit_registered:
                try:
                    atexit.register(self._on_exit)
                    self._atexit_registered = True
                except Exception:
                    pass
            if self._thread is None or not self._thread.is_alive():
                try:
                    self._thread = threading.Thread(
                        target=self._run, name="loopgain-funnel", daemon=True
                    )
                    self._thread.start()
                except Exception:
                    self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.FLUSH_INTERVAL):
            self.flush_now()

    def flush_now(self) -> bool:
        """Drain the queue and POST it as one batch. Never raises.

        Returns ``True`` if a batch was sent and acknowledged (2xx), ``False``
        otherwise (including when there was nothing to send).
        """
        with self._lock:
            if not self._queue:
                return False
            pending = self._queue
            self._queue = []
        batch = {
            "schema_version": FUNNEL_SCHEMA_VERSION,
            "library": "loopgain",
            "events": pending,
        }
        try:
            return bool(self._sender(self._endpoint, batch, self.SEND_TIMEOUT))
        except Exception:
            return False

    def _on_exit(self) -> None:
        # Emit the recurring-use "session" summary, then make a final flush
        # attempt. Bounded so a hung socket can't delay interpreter shutdown.
        try:
            self._emit_session_summary()
        except Exception:
            pass
        self._stop.set()
        try:
            self.flush_now()
        except Exception:
            pass
        t = self._thread
        if t is not None and t.is_alive():
            try:
                t.join(timeout=self.SEND_TIMEOUT + 0.5)
            except Exception:
                pass

    def _emit_session_summary(self) -> None:
        if self._mode != _ENABLED or not self._session_started:
            return
        event = self._base_event("session")
        event["session_seq"] = int(self._state.get("session_count", 0))
        event["adapter"] = self._adapter
        if self._outcomes:
            event["outcomes"] = dict(self._outcomes)
        self._enqueue(event)

    # ----- Public hooks (called from core / adapters) -----

    def on_init(self) -> None:
        """A ``LoopGain`` was constructed. Records first_init once per install."""
        mode = self._ensure_loaded()
        if mode != _ENABLED:
            return
        with self._lock:
            if not self._state.get("first_init_at"):
                self._state["first_init_at"] = _now_hour(self._clock)
                self._write_state_file()
                self._enqueue(self._base_event("first_init"))
            else:
                # Still need the flush machinery armed so the session summary
                # at exit is delivered even on installs past the first.
                self._ensure_thread()

    def on_first_observe(self) -> None:
        """``observe()`` was called for the first time ever (activation)."""
        if self._ensure_loaded() != _ENABLED:
            return
        with self._lock:
            if self._state.get("first_observe_at"):
                return
            self._state["first_observe_at"] = _now_hour(self._clock)
            self._write_state_file()
        self._enqueue(self._base_event("first_observe"))

    def note_outcome(self, state: str) -> None:
        """Record a terminal loop state into the coarse outcome distribution."""
        if self._mode != _ENABLED:
            return
        bucket = _STATE_TO_OUTCOME.get(state, "other")
        with self._lock:
            self._outcomes[bucket] = self._outcomes.get(bucket, 0) + 1

    def note_adapter(self, name: Optional[str]) -> None:
        """Record which framework adapter is driving the loop."""
        if self._mode != _ENABLED or not name:
            return
        self._adapter = str(name)

    # ----- CLI support -----

    def set_consent(self, granted: bool) -> None:
        """Persist an explicit opt-in / opt-out decision (used by the CLI)."""
        with self._lock:
            self._state = self._read_state_file()
            if not self._state.get("instance_id"):
                self._state["instance_id"] = uuid.uuid4().hex
            self._state["schema"] = FUNNEL_SCHEMA_VERSION
            self._state["consent"] = "granted" if granted else "denied"
            # A decision means the notice is moot.
            self._state["notice_shown"] = True
            self._write_state_file()

    def reset(self) -> None:
        """Forget the instance id and consent (regenerated on next opt-in)."""
        with self._lock:
            try:
                if os.path.exists(self._state_path):
                    os.remove(self._state_path)
            except Exception:
                pass
            self._state = {}
            self._loaded = False
            self._mode = None

    def status(self) -> dict[str, Any]:
        """A snapshot for ``loopgain telemetry --show``. Reads, never sends."""
        state = self._read_state_file()
        mode = self._resolve_mode()
        consent_source = "default (undecided)"
        env = self._env_consent()
        if env is True:
            consent_source = "environment (LOOPGAIN_TELEMETRY)"
        elif env is False:
            if os.environ.get("DO_NOT_TRACK", "").strip().lower() in _TRUE:
                consent_source = "environment (DO_NOT_TRACK)"
            else:
                consent_source = "environment (LOOPGAIN_TELEMETRY)"
        elif state.get("consent") in ("granted", "denied"):
            consent_source = "config file"
        elif self._is_ci():
            consent_source = "CI detected (declined)"
        return {
            "enabled": mode == _ENABLED,
            "mode": mode,
            "consent_source": consent_source,
            "instance_id": state.get("instance_id"),
            "endpoint": self._endpoint,
            "config_file": self._state_path,
            "first_init_at": state.get("first_init_at"),
            "first_observe_at": state.get("first_observe_at"),
            "session_count": state.get("session_count", 0),
        }


# ----- Module singleton + fail-silent hook wrappers -----
#
# core.py and the adapters call the module-level functions below, never the
# class directly. Every wrapper swallows all exceptions: funnel telemetry can
# never raise into a user's loop.

_INSTANCE: Optional[Funnel] = None
_INSTANCE_LOCK = threading.Lock()


def _instance() -> Funnel:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = Funnel()
    return _INSTANCE


def on_init() -> None:
    try:
        _instance().on_init()
    except Exception:
        pass


def on_first_observe() -> None:
    try:
        _instance().on_first_observe()
    except Exception:
        pass


def note_outcome(state: str) -> None:
    try:
        _instance().note_outcome(state)
    except Exception:
        pass


def note_adapter(name: Optional[str]) -> None:
    try:
        _instance().note_adapter(name)
    except Exception:
        pass
