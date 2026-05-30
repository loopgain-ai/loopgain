# Telemetry in `loopgain`

LoopGain has **two completely separate** telemetry paths. They serve
different purposes, are configured differently, and never share data or code.
This document is about the second one — **anonymous funnel telemetry** — and
how to control it.

| | Product telemetry (your loop data) | Funnel telemetry (anonymous usage) |
|---|---|---|
| **Module** | `loopgain.telemetry` | `loopgain.funnel` |
| **Triggered by** | you calling `LoopGain.send_telemetry(...)` | importing/using the library |
| **What it's for** | sending *your* loop's Aβ traces to *your* dashboard | telling the maintainer whether the project is being used |
| **Auth** | your bearer token | none |
| **Volume** | one POST per loop you choose to report | a few events per install |
| **Default** | off (you must call it) | **off (opt-in, default-decline)** |
| **Contains** | your loop statistics (never prompts/outputs) | anonymous counts only |

If you came here about sending *your own* loop data to a dashboard, that's
the product receiver — see the `send_telemetry` section in the README, not
this file.

---

## Funnel telemetry: opt-in, default-decline

The open-source library is otherwise invisible to its maintainer: there's no
server in the loop, so there's no way to know if anyone installs it, gets it
working, or keeps using it. Funnel telemetry exists to answer exactly that —
**install → first `observe()` → recurring use** — and nothing more.

**It is off by default and sends nothing unless you explicitly opt in.** On
first use you'll see a one-time notice in your terminal explaining this. We do
not phone home on a default install.

### Turning it on or off

```bash
# Opt in (thank you — this genuinely helps a solo project):
loopgain telemetry --enable
#   ...or, per-shell / per-CI:
export LOOPGAIN_TELEMETRY=1

# Opt out explicitly (also the default if you do nothing):
loopgain telemetry --disable
export LOOPGAIN_TELEMETRY=0

# The standard DO_NOT_TRACK convention is honored as a hard opt-out:
export DO_NOT_TRACK=1

# See current status and exactly what would be sent:
loopgain telemetry --show

# Forget the local instance id + your choice:
loopgain telemetry --reset
```

Resolution order: `DO_NOT_TRACK` → `LOOPGAIN_TELEMETRY` → your saved choice
(`loopgain telemetry --enable/--disable`) → **declined by default**. CI
environments (`CI`, `GITHUB_ACTIONS`, …) are auto-detected and declined
silently — no notice, no data.

### What is collected (only when you opt in)

- A **random instance id** — a fresh `uuid4` generated and stored locally. It
  is **not** derived from your hardware, username, hostname, MAC address, or
  any other identifier. It exists only to avoid counting one install as many.
- **Hour-bucketed timestamps** for three funnel events: first init, first
  `observe()`, and each session.
- **`loopgain` version, Python version (major.minor), and OS family**
  (`Darwin` / `Linux` / `Windows`).
- **Which framework adapter** drove the loop (`langgraph`, `crewai`, …), if any.
- A **coarse count of loop outcomes** (`converged` / `oscillating` /
  `diverged` / `stalled` / `max_iterations` / `other`).

### What is *never* collected

- ❌ Prompts, completions, or any model input/output
- ❌ Error contents — only counts and coarse outcome buckets
- ❌ API keys, tokens, or credentials of any kind
- ❌ File paths, working directories, repo names, or environment variables
- ❌ IP addresses, hostnames, usernames, or any identity
- ❌ Individual Aβ values or error magnitudes

The privacy contract is enforced by unit tests in `tests/test_funnel.py`
(including a test that runs a real loop carrying secret data and asserts none
of it appears in any funnel payload).

### How it's delivered

Events are queued in memory and flushed in **batches** by a background daemon
thread and on interpreter exit. Delivery is **https-only**, short-timeout, and
**completely fail-silent**: any error — network down, bad config, a bug in the
telemetry code itself — is swallowed. Funnel telemetry can never raise into,
slow down, or break your loop. The state file lives at
`$XDG_CONFIG_HOME/loopgain/funnel.json` (default `~/.config/loopgain/`).

### Where it goes

The default endpoint is `https://telemetry.loopgain.ai/v1/funnel` — a
different route from the product receiver's `/v1/aggregate`. Override it with
`LOOPGAIN_TELEMETRY_FUNNEL_ENDPOINT` if you self-host.

---

*Questions or concerns about telemetry? Open an issue at
<https://github.com/loopgain-ai/loopgain/issues>. Getting this right matters
to us.*
