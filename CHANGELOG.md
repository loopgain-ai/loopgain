# Changelog

All notable changes to the `loopgain` library are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-05-18

### Changed
- **New default classifier: multi-feature trajectory classifier.** The
  v0.1 single-Aβ-band classifier (thresholds 0.3 / 0.85 / 0.95 / 1.05)
  has been replaced as the default by a trajectory classifier that reads
  four features off the full error history: cumulative `E_ratio`,
  log-domain OLS `slope_log`, slope-significance `slope_p` (Student-t
  two-sided), and detrended residual std `osc_std`. The five state
  names (`FAST_CONVERGE`, `CONVERGING`, `STALLING`, `OSCILLATING`,
  `DIVERGING`) are unchanged. Pre-registered in
  `PROTOCOL_v2_classifier.md`; validated at 98.8% macro-averaged
  accuracy on N=1000 deterministic-mock trajectories
  (`RESULTS_v2_classifier.md`). Motivation: the v0.1 classifier scored
  37.3% accuracy on the Component Algebra Lab v1 Experiment 3
  (2026-04-10, 150 real-LLM GVR loops) because a single instantaneous
  Aβ value cannot disambiguate floor-noise convergence, slow
  monotone improvement, and mild drift-style divergence.
- **`STALLING` is now terminal after 2 consecutive readings** (v2
  protocol §3.3, "Return best-so-far"). Surfaced as a new
  `outcome="stalled"` distinct from `oscillating`. The dashboard's
  `bandFromEvent` routes it to the `STALLING` band.
- **Legacy classifier preserved** via `LoopGain(classifier='legacy_bands')`
  for callers that have empirically tuned `ThresholdBands` against a
  specific workload.

### Added
- `loopgain.classifier` module exposing `TrajectoryThresholds`,
  `TrajectoryFeatures`, `extract_features`, and `classify_trajectory` for
  post-hoc classification of stored error histories.
- New `trajectory_thresholds` and `classifier` keyword arguments on
  `LoopGain.__init__`.
- `tests/test_classifier_synthetic.py` (27 tests, math-correctness gate).
- `tests/test_classifier_mock_validation.py` (12 tests, deterministic-mock
  validation at N=200 per regime).
- `PROTOCOL_v2_classifier.md` — pre-registered design + threshold
  derivations + validation plan.
- `RESULTS_v2_classifier.md` — full validation report (synthetic, Tier-2
  re-classification of the 150 v1 trials, Tier-3 30-trial real-LLM
  confirmatory, Tier-A 1000-trial deterministic-mock).

### Added (framework adapters, unchanged from earlier in 0.2.0 cycle)
- **Three new framework adapters.** LoopGain now ships pre-built
  integrations for six major agent frameworks (up from three):
  - **LangChain** (`pip install 'loopgain[langchain]'`) — duck-types
    against any `langchain.agents.create_agent()` result (v1+) or the
    legacy `AgentExecutor`. Forwards `**stream_kwargs` verbatim so the
    user controls chunk shape. Sync + async paths.
  - **OpenAI Agents SDK** (`pip install 'loopgain[openai-agents]'`) —
    wraps `Runner.run_streamed(agent, input).stream_events()`. Async-first
    with a `run_sync` helper. Default observation filter limits
    `error_fn` to `run_item_stream_event` (override via
    `observe_event_types`). Best-effort calls `.cancel()` on the
    underlying `RunResultStreaming` at terminal state.
  - **Claude Agent SDK** (`pip install 'loopgain[claude-agent-sdk]'`) —
    wraps `claude_agent_sdk.query(prompt=..., options=...)`. Accepts
    either a `prompt` (constructs the iterator internally) or a
    pre-built `message_iterator` (for `ClaudeSDKClient.receive_messages()`
    callers). Default observation filter limits `error_fn` to
    `AssistantMessage` (override via `observe_message_types`).
- Real-framework integration smoke tests for each new adapter
  (skipped if the framework isn't installed in the test environment).
- README adapter section grew to six stanzas; landing page (loopgain.ai)
  updated in lockstep.

## [0.1.9] — 2026-05-16

### Changed
- **Breaking**: `target_error=0.0` (still the default) now short-circuits
  the loop on exactly-zero error — the natural completion signal for
  verifier-driven loops (no failing tests, no validation errors). The
  previous semantics — "`target_error=0.0` disables the short-circuit" —
  conflicted with users' reading of the parameter name and required the
  awkward `target_error=0.5` workaround when "zero failures = done" was
  the intent. To disable the short-circuit entirely, pass
  `target_error=None`. The parameter type is now `Optional[float]`.

### Added
- `examples/` directory with runnable end-to-end scripts demonstrating
  LoopGain on real Anthropic Claude calls. `01_code_pytest.py` is fully
  implemented; `02`-`06` are documented stubs targeting one stability
  band each. Optional install: `pip install 'loopgain[examples]'`.
- `Makefile` with manual `make examples` target (no CI integration —
  scripts spend real API budget when invoked).
- `CHANGELOG.md` (this file).

### Fixed
- `build_telemetry_payload` now coerces non-finite floats (`inf`, `-inf`,
  `NaN`) to `None` before serialization. Previously, a constant-error
  trajectory pushed `gain_margin = 1/max(Aβ) = +inf`, which `json.dumps`
  emits as the literal token `Infinity` — invalid per RFC 8259 — causing
  strict receivers (including `telemetry.loopgain.ai`) to reject the
  payload with HTTP 400 and `send_telemetry` to return `False`. The
  receiver now sees a strict-JSON `null` and renders "no data" instead.

## [0.1.8] — 2026-05-15

### Changed
- Documentation-only release. README rewritten for accuracy: the
  cloud-aggregator surface (`telemetry.loopgain.ai` + `dashboard.loopgain.ai`)
  is now correctly described as live separate repos rather than future
  roadmap. No code changes.

## [0.1.7] — 2026-05-14

### Added
- Framework adapter extras for LangGraph, CrewAI, and AutoGen v0.4+,
  installable via `pip install 'loopgain[langgraph]'` etc. Adapters
  auto-stamp `framework=<name>` on outbound telemetry.
- Real-framework integration smoke tests for each adapter (skipped if
  the framework isn't installed in the test environment).

### Security
- `send_telemetry` now refuses to follow HTTP 3xx redirects, preventing
  a compromised endpoint from harvesting the bearer token via a
  cross-origin 302.

### Changed
- `__version__` is now sourced from a single `loopgain/_version.py`,
  eliminating drift between `pyproject.toml` and `loopgain.__version__`.

## [0.1.6] — 2026-05-14

### Added
- Telemetry schema v3: optional `per_iteration` block carrying capped
  Aβ-smoothed and error-magnitude trajectories (256 entries max) plus
  classification fields (`framework`, `loop_type`, `team`) to drive
  dashboard filters. Receiver remains backward-compatible with v1/v2
  payloads.

## [0.1.5] — 2026-05-13

### Security
- `send_telemetry` rejects non-`https://` endpoints by default; HTTP is
  opt-in via `allow_insecure=True` and intended only for self-hosted
  receivers on localhost.

## [0.1.4] — 2026-05-13

### Added
- Telemetry schema v2: `first_eta_prediction` and `first_eta_at_iteration`
  captured for the dashboard's ETA Accuracy panel.

## [0.1.3] — 2026-05-12

### Changed
- README accuracy pass.

## [0.1.2] — 2026-05-12

### Changed
- Public README scrubbed of internal-tooling references.

## [0.1.1] — 2026-05-12

### Fixed
- First clean PyPI upload — 0.1.0 was a stale build artifact.

## [0.1.0] — 2026-05-12

### Added
- Initial public release. Five-band Barkhausen stability monitor with
  EMA-smoothed loop gain, best-so-far rollback, ETA prediction, and
  opt-in anonymized telemetry hook. Pure Python, zero runtime
  dependencies, Python 3.10+.
