"""Tests for opt-in anonymous funnel telemetry (loopgain.funnel).

This module is distinct from the product receiver tested in
test_telemetry.py. The contract under test here:

* **Default-decline.** Nothing is sent unless the user explicitly opts in.
* **Privacy.** When enabled, payloads carry only anonymous counts/metadata —
  never prompts, outputs, error contents, keys, paths, or IPs.
* **Fail-silent.** A funnel bug or network failure never raises into the loop.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

import loopgain.funnel as funnel_mod
from loopgain.funnel import Funnel


FIXED_CLOCK = lambda: datetime(2026, 5, 30, 9, 37, 12, 500, tzinfo=timezone.utc)


class _Capture:
    """A fake sender that records every batch instead of doing network I/O."""

    def __init__(self, ok: bool = True):
        self.batches: list[dict] = []
        self.ok = ok

    def __call__(self, endpoint, batch, timeout):
        self.batches.append(batch)
        return self.ok

    @property
    def events(self):
        out = []
        for b in self.batches:
            out.extend(b.get("events", []))
        return out


def _funnel(tmp_path, sender, **kw) -> Funnel:
    return Funnel(
        config_dir=str(tmp_path),
        sender=sender,
        clock=FIXED_CLOCK,
        start_background=False,
        **kw,
    )


# ----- Consent resolution -----


def test_disabled_via_env_writes_nothing_and_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOPGAIN_TELEMETRY", "0")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    cap = _Capture()
    f = _funnel(tmp_path, cap)
    f.on_init()
    f.on_first_observe()
    f.flush_now()
    assert cap.batches == []
    assert not os.path.exists(os.path.join(str(tmp_path), "funnel.json"))


def test_do_not_track_overrides_explicit_enable(tmp_path, monkeypatch):
    """DO_NOT_TRACK=1 is a hard opt-out even if LOOPGAIN_TELEMETRY=1."""
    monkeypatch.setenv("LOOPGAIN_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    cap = _Capture()
    f = _funnel(tmp_path, cap)
    f.on_init()
    f.flush_now()
    assert cap.batches == []


def test_undecided_is_default_decline_no_send_but_writes_state(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOPGAIN_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    for k in ("CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI"):
        monkeypatch.delenv(k, raising=False)
    cap = _Capture()
    f = _funnel(tmp_path, cap)
    f.on_init()
    f.on_first_observe()
    f.flush_now()
    # Default-decline: nothing sent...
    assert cap.batches == []
    # ...but a state file with an instance id + notice marker exists.
    state = json.loads((tmp_path / "funnel.json").read_text())
    assert state["instance_id"]
    assert state["notice_shown"] is True
    assert "first_init_at" not in state  # no funnel events recorded when undecided


def test_ci_environment_declines_silently(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOPGAIN_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setenv("CI", "true")
    cap = _Capture()
    f = _funnel(tmp_path, cap)
    f.on_init()
    f.flush_now()
    assert cap.batches == []


@pytest.mark.parametrize("val", ["1", "true", "YES", "on", "Enabled"])
def test_enabled_spellings(tmp_path, monkeypatch, val):
    monkeypatch.setenv("LOOPGAIN_TELEMETRY", val)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    cap = _Capture()
    f = _funnel(tmp_path, cap)
    f.on_init()
    f.flush_now()
    assert len(cap.events) == 1
    assert cap.events[0]["event"] == "first_init"


# ----- Enabled: funnel events -----


def _enabled(tmp_path, monkeypatch, cap, **kw):
    monkeypatch.setenv("LOOPGAIN_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    return _funnel(tmp_path, cap, **kw)


def test_first_init_and_first_observe_emitted_once(tmp_path, monkeypatch):
    cap = _Capture()
    f = _enabled(tmp_path, monkeypatch, cap)
    f.on_init()
    f.on_first_observe()
    # Repeat calls must not re-emit.
    f.on_init()
    f.on_first_observe()
    f.flush_now()
    names = [e["event"] for e in cap.events]
    assert names.count("first_init") == 1
    assert names.count("first_observe") == 1


def test_first_init_not_re_emitted_across_processes(tmp_path, monkeypatch):
    """A second 'install run' (fresh Funnel, same config dir) does not re-emit
    first_init — the funnel start is recorded once per install."""
    cap1 = _Capture()
    f1 = _enabled(tmp_path, monkeypatch, cap1)
    f1.on_init()
    f1.flush_now()
    assert [e["event"] for e in cap1.events] == ["first_init"]

    cap2 = _Capture()
    f2 = _enabled(tmp_path, monkeypatch, cap2)
    f2.on_init()
    f2.flush_now()
    assert "first_init" not in [e["event"] for e in cap2.events]


def test_instance_id_stable_across_reload(tmp_path, monkeypatch):
    cap = _Capture()
    f1 = _enabled(tmp_path, monkeypatch, cap)
    f1.on_init()
    f1.flush_now()
    id1 = cap.events[0]["instance_id"]
    cap2 = _Capture()
    f2 = _enabled(tmp_path, monkeypatch, cap2)
    f2.on_init()  # no new first_init, but instance id should be the same
    assert json.loads((tmp_path / "funnel.json").read_text())["instance_id"] == id1


def test_event_payload_shape(tmp_path, monkeypatch):
    cap = _Capture()
    f = _enabled(tmp_path, monkeypatch, cap)
    f.on_init()
    f.flush_now()
    ev = cap.events[0]
    assert ev["event"] == "first_init"
    assert isinstance(ev["instance_id"], str) and len(ev["instance_id"]) == 32
    assert "library" not in ev  # "library" lives on the batch envelope, not events
    assert ev["library_version"]
    assert "." in ev["python"]
    assert ev["os"]
    # Hour-bucketed timestamp — no minute/second.
    assert ev["ts_hour"].startswith("2026-05-30T09:00:00")
    assert "37" not in ev["ts_hour"].split("T")[1]


def test_batch_envelope_shape(tmp_path, monkeypatch):
    cap = _Capture()
    f = _enabled(tmp_path, monkeypatch, cap)
    f.on_init()
    f.flush_now()
    batch = cap.batches[0]
    assert batch["library"] == "loopgain"
    assert batch["schema_version"] == funnel_mod.FUNNEL_SCHEMA_VERSION
    assert isinstance(batch["events"], list)


def test_session_summary_carries_outcomes_and_adapter(tmp_path, monkeypatch):
    cap = _Capture()
    f = _enabled(tmp_path, monkeypatch, cap)
    f.on_init()
    f.note_adapter("langgraph")
    f.note_outcome("DIVERGING")
    f.note_outcome("DIVERGING")
    f.note_outcome("TARGET_MET")
    f.note_outcome("SOMETHING_NEW")  # unknown → "other"
    f._emit_session_summary()
    f.flush_now()
    session = [e for e in cap.events if e["event"] == "session"]
    assert len(session) == 1
    s = session[0]
    assert s["adapter"] == "langgraph"
    assert s["session_seq"] == 1
    assert s["outcomes"] == {"diverged": 2, "converged": 1, "other": 1}


# ----- Privacy contract -----


def test_no_forbidden_content_in_funnel_payload(tmp_path, monkeypatch):
    """Even with sensitive data flowing through a real loop, the funnel
    payloads must contain none of it."""
    cap = _Capture()
    f = _enabled(tmp_path, monkeypatch, cap)
    # Inject the funnel instance so a real LoopGain run routes through it.
    monkeypatch.setattr(funnel_mod, "_INSTANCE", f)

    from loopgain import LoopGain

    secret = "SUPER-SECRET-PROMPT-AND-OUTPUT"
    lg = LoopGain(target_error=0.5, max_iterations=10)
    for e in [10.0, 4.0, 0.3]:
        if not lg.should_continue():
            break
        lg.observe(e, output={"prompt": secret, "completion": secret})
    f._emit_session_summary()
    f.flush_now()

    blob = json.dumps(cap.batches)
    assert secret not in blob
    assert "prompt" not in blob
    assert "completion" not in blob
    # Sanity: we DID capture real funnel events from the run.
    names = {e["event"] for e in cap.events}
    assert "first_init" in names and "first_observe" in names


# ----- Fail-silent -----


def test_flush_swallows_sender_exception(tmp_path, monkeypatch):
    def boom(endpoint, batch, timeout):
        raise RuntimeError("network on fire")

    f = _enabled(tmp_path, monkeypatch, boom)
    f.on_init()
    # Must not raise.
    assert f.flush_now() is False


def test_real_unreachable_endpoint_returns_false(tmp_path, monkeypatch):
    """The default sender against a dead endpoint fails quietly."""
    monkeypatch.setenv("LOOPGAIN_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    f = Funnel(
        config_dir=str(tmp_path),
        endpoint="https://127.0.0.1:1/v1/funnel",  # nothing listens
        clock=FIXED_CLOCK,
        start_background=False,
    )
    f.SEND_TIMEOUT = 0.5
    f.on_init()
    assert f.flush_now() is False


def test_default_sender_rejects_non_https(tmp_path, monkeypatch):
    """Funnel events never go out over http:// (no scheme downgrade)."""
    sent = {"n": 0}

    def fake_open(req, timeout):
        sent["n"] += 1
        raise AssertionError("must not open a non-https request")

    monkeypatch.setattr(funnel_mod, "_open_request", fake_open)
    ok = funnel_mod._default_sender("http://telemetry.loopgain.ai/v1/funnel", {"events": []}, 1.0)
    assert ok is False
    assert sent["n"] == 0


def test_module_hooks_never_raise(monkeypatch):
    """The module-level hook wrappers swallow everything, even if the
    singleton is wedged."""

    class Boom:
        def on_init(self):
            raise RuntimeError("x")

        def on_first_observe(self):
            raise RuntimeError("x")

        def note_outcome(self, s):
            raise RuntimeError("x")

        def note_adapter(self, n):
            raise RuntimeError("x")

    monkeypatch.setattr(funnel_mod, "_INSTANCE", Boom())
    # None of these may raise.
    funnel_mod.on_init()
    funnel_mod.on_first_observe()
    funnel_mod.note_outcome("DIVERGING")
    funnel_mod.note_adapter("langgraph")


# ----- CLI control surface -----


def test_set_consent_and_status_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOPGAIN_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    f = _funnel(tmp_path, _Capture())
    f.set_consent(True)
    assert Funnel(config_dir=str(tmp_path), start_background=False).status()["enabled"] is True
    f.set_consent(False)
    assert Funnel(config_dir=str(tmp_path), start_background=False).status()["enabled"] is False


def test_reset_forgets_state(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOPGAIN_TELEMETRY", raising=False)
    f = _funnel(tmp_path, _Capture())
    f.set_consent(True)
    assert os.path.exists(os.path.join(str(tmp_path), "funnel.json"))
    f.reset()
    assert not os.path.exists(os.path.join(str(tmp_path), "funnel.json"))


def test_cli_enable_disable_show(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOOPGAIN_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOOPGAIN_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    from loopgain import cli

    assert cli.main(["telemetry", "--enable"]) == 0
    capsys.readouterr()
    assert cli.main(["telemetry", "--show"]) == 0
    assert "ON" in capsys.readouterr().out

    assert cli.main(["telemetry", "--disable"]) == 0
    capsys.readouterr()
    assert cli.main(["telemetry", "--show"]) == 0
    assert "OFF" in capsys.readouterr().out


def test_cli_version_and_no_command(tmp_path, monkeypatch, capsys):
    from loopgain import cli
    from loopgain import __version__

    assert cli.main(["version"]) == 0
    assert __version__ in capsys.readouterr().out
    # No subcommand prints help and returns non-zero.
    assert cli.main([]) == 1
