# Changelog

All notable changes to the `loopgain` library are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.2] — 2026-07-07

Documentation-only patch. No library, API, or behaviour change from 0.6.1.

- **Synced the PyPI page and README to the 0.6.1 `send_telemetry` signature.**
  `actual_dollars_spent`/`actual_dollars_saved` shipped in code in 0.6.1 but
  the README's documented signature and the PyPI long description (built
  from the README at publish time) still showed the pre-0.6.1 parameter
  list.
- **Made the measured-only contract explicit in the docstrings and README.**
  `actual_dollars_spent`/`actual_dollars_saved` must be a genuinely measured
  quantity — summed real API usage x list price, or an actually-executed
  paired-baseline run — never a formula-derived estimate. Written down after
  nearly wiring a chars/4 token-estimate-based counterfactual into
  `actual_dollars_saved` in a downstream integration; the dashboard trusts
  these fields as ground truth and stops extrapolating once populated, so an
  estimate passed through them would silently degrade that guarantee for
  every consumer, not just the caller.

## [0.6.1] — 2026-07-07

Additive, backward-compatible feature. Default behaviour is byte-identical to
0.6.0 when the new parameters are left unset.

- **`send_telemetry(actual_dollars_spent=…, actual_dollars_saved=…)`.**
  The receiver's `loop_events` table has carried first-class
  `actual_dollars_spent` / `actual_dollars_saved` columns since v3.1/v3.2
  (2026-05-25/26) and the dashboard already prefers them over its
  iter-count x $/iter extrapolation whenever present — but the SDK never
  exposed a way to populate them, so callers with real per-run cost data
  (summed token usage x list price, or a measured paired-baseline delta)
  had no supported path except smuggling numbers into the opaque `team`
  label. Both are now optional keyword arguments on `send_telemetry()`,
  threaded straight through `build_payload()`. Omit either (or both) for
  unchanged behaviour.

## [0.6.0] — 2026-06-12

Additive, backward-compatible feature. Default behaviour is byte-identical to
0.5.2 when the new parameter is left unset.

- **Configurable consecutive-STALLING kill — `LoopGain(stall_terminate_count=…)`.**
  The trajectory classifier terminates a stalled loop after N *consecutive*
  STALLING readings. That count was hardcoded to 2; it is now a constructor
  parameter, default `2`, so existing loops are unchanged. The default is tuned
  for inner / per-generation loops where a brief plateau is a reliable stop
  signal. Session-scale / outer loops (e.g. Ralph-style runs where each
  iteration is a whole agent session) should raise it — a single
  regression-then-recovery session reads as a transient stall, and the
  impatient default-2 kill stops too early, discarding a better answer that
  arrives just after. The 2026-06-11 outer-loop study found that raising the
  count retained every true catch while roughly halving false stops (consensus
  best value ≈ 5; the exact session-scale default is not yet statistically
  pinned — a separate ~1000-run study is pending, so the library ships the
  conservative inner-loop default and exposes the knob). This is distinct from
  `TrajectoryThresholds.stall_patience`, which governs STALLING *onset*, not how
  many consecutive STALLING labels terminate the loop. Ignored under
  `classifier="legacy_bands"`.

## [0.5.2] — 2026-06-10

Documentation-only patch. No library, API, or behaviour change from 0.5.1.

- **Privacy contract corrected to match the code.** The README's telemetry
  section previously said individual Aβ values are never sent — but with the
  default `include_per_iteration=True`, `send_telemetry` includes a
  length-capped per-iteration trajectory (smoothed Aβ values and numeric error
  magnitudes; it drives the dashboard's convergence-profile scrubbing). The
  README now discloses this and notes `include_per_iteration=False` sends the
  aggregate summary only. What is never sent is unchanged: prompts,
  completions, error contents, the output buffer, or any customer identity
  beyond the bearer token.
- Honest-limits bullet now carries the measured wrong-fixed-point rate
  (4.5% of converged runs on the benchmark's code-gen workload), with the
  false-stop rate disambiguated from it.
- Removed a leftover ETA-era phrase from the Background section; "median cut"
  corrected to "total" for the 92.8% benchmark claim.

## [0.5.1] — 2026-06-09

Documentation-only patch. No library, API, or behaviour change from 0.5.0.

- Synced the PyPI project page with the current GitHub README (the value-first
  "cost controller" rewrite); the 0.5.0 page predated it.
- Dropped a stale "gain margin" mention from the telemetry-payload docstring
  (the field itself was already removed from the payload in 0.5.0 / schema v4).

## [0.5.0] — 2026-06-09

Removed two derived signals — **ETA prediction** and **gain margin** — that did
not hold up on real agent loops. Both were tested against the public benchmark
and neither earned its place: the closed-form ETA is ill-posed at `target_error
= 0` (the dominant real task class) and fired on a tiny fraction of trajectories;
gain margin (`1 / max(Aβ_smooth)`) is undefined for the majority of loops that
converge in a single iteration, so it could only ever describe the struggling
minority. Rather than ship signals we don't trust, they're gone from every
public surface. The reasoning is preserved in the benchmark `LESSONS.md`.

This is a **breaking change** to the public result object and the telemetry
payload, hence the minor-version bump.

### Removed
- **`LoopGainResult.gain_margin`**, the **`LoopGain.gain_margin`** property, and
  the **`LoopGain.eta`** property are removed. The underlying Aβ trajectory is
  unaffected — `result.convergence_profile` (the smoothed loop-gain values) and
  the five-band classifier are unchanged.
- **`LoopGainResult.first_eta_prediction`** and
  **`LoopGainResult.first_eta_at_iteration`** are removed.
- The telemetry payload no longer carries `gain_margin`,
  `first_eta_prediction`, or `first_eta_at_iteration`. **Schema bumped to v4.**
  The receiver still accepts older payloads; the dropped fields are ignored.

## [0.4.3] — 2026-06-08

Telemetry delivery reliability. Best-effort sends now survive a transient blip
instead of silently dropping. Backward-compatible; additive parameters only,
no public API change.

### Changed
- **`send_payload` / `LoopGain.send_telemetry` now retry transient failures.**
  The warm round-trip to the receiver is ~150 ms, well inside the 2 s timeout,
  but a transient outlier (a cold database first-write, a momentary network
  blip) that blew past it was previously dropped with no retry — and a caller
  that sends one aggregate per run would lose that whole run's data. Sends now
  retry up to 2 times (3 attempts total) with a short linear backoff (0.25 s,
  0.50 s) on *transient* failures only — timeouts, connection errors, and
  `5xx`/`429` responses. Deterministic failures (`4xx` such as a bad token, a
  malformed payload, a refused redirect) are **not** retried. Still fully
  best-effort: the send path never raises and can never break the caller's loop.

### Added
- **`retries` and `retry_backoff` parameters** on `send_payload` and
  `LoopGain.send_telemetry` (defaults `2` and `0.25`). Set `retries=0` to
  restore the previous single-attempt behavior.

## [0.4.2] — 2026-06-05

Correctness + telemetry. A statistics fix to the trajectory classifier (no
effect on any published benchmark number — see below) plus one additive
telemetry field. Backward-compatible; no public API change.

### Fixed
- **Corrected the df=2 (n=4) two-sided t-test p-value.** The `df=2` branch of
  `_two_sided_t_p` returned a p-value exactly 2× too large, requiring
  `|t| > 6.21` for significance instead of the correct `|t| > 4.30`. This made
  the classifier too conservative on 4-iteration trajectories, mislabeling
  ~11% of *marginal* n=4 converging loops as `STALLING` instead of
  `CONVERGING`. Now uses the exact closed form `1 − |t|/√(2 + t²)`; pinned by
  exact-value regression tests (`test_two_sided_t_p_df1/df2_exact`). The df=1
  and df≥3 branches were already correct. **Bench impact: none** — of 2,000
  benchmark trials only 21 reach iteration 4 and zero fall in the affected
  marginal band, so the published distribution and cost numbers are unchanged
  (verified by controlled old-vs-new replay over the recorded trajectories).

### Added
- **`best_index` in the loop telemetry payload** — the 0-based index of the
  lowest-error iteration. Lets the receiver derive iterations-to-best and
  iterations-past-best for the Iteration-Waste view. Privacy-safe (an integer
  position); the ingest path is otherwise unchanged.

### Changed
- Test badge now reads "200+ passing" instead of a hard count. The collected
  test total is adapter-dependent (which framework extras are installed changes
  collection), so a fixed number drifts; "200+" is the honest, stable claim.

## [0.4.1] — 2026-06-04

Docs / packaging only — **no library code change** (0.4.0's runtime behavior is
unchanged). This release exists solely to update the immutable PyPI metadata,
which can't be edited in place once published.

### Changed
- **Product descriptor is now "cost controller," not "stability monitor."** The
  README headline and the PyPI package description now lead with cost control
  ("an open-source cost controller for AI agent loops"), matching the rest of
  the public surface. The Barkhausen stability *criterion* (the `Aβ` loop-gain
  math) stays as the technical "how" in the body — only the product descriptor
  changed.
- **Dropped "ETA prediction" from the package summary.** The closed-form ETA is
  still shipped (`lg.eta`), but it fires on too few real trajectories to earn a
  headline mention; removed from the one-line description to avoid over-claiming.
- Test badge corrected 157 → 202 passing.

## [0.4.0] — 2026-06-03

### Fixed
- **Loops could run unbounded on a plateau (liveness bug).** The trajectory
  classifier emitted its "continue" verdicts (`FAST_CONVERGE`, `CONVERGING`)
  from *cumulative* error reduction and a *whole-history* slope — both
  describe the past and never expire. A loop that reduced its error and then
  plateaued or oscillated *below* the cumulative threshold stayed pinned in a
  continue-state, never reached `STALLING`/`OSCILLATING`, and — with the old
  default `max_iterations=None` — never terminated (`should_continue()`
  returned `True` forever). Output was never wrong (best-so-far rollback held
  the good answer) but the loop never returned to hand it back. Fixed with a
  liveness gate: the continue-verdicts are now withdrawn once a loop has gone
  `stall_patience` iterations (default 3) without achieving a new lowest
  error, so a stalled/oscillating loop terminates and returns best-so-far.
  **Action for users:** none required — the fix is automatic. If you relied on
  `max_iterations=None` + `target_error=None` and saw loops hang on a plateau,
  upgrade.

### Changed
- **`max_iterations` now defaults to `50`** (was `None`) as a hard safety
  backstop, so the library can never run truly unbounded even if a loop never
  converges and never stalls. A stability verdict normally terminates the loop
  long before 50. Pass `max_iterations=None` to restore the old fully-unbounded
  behavior, or a smaller integer to cap tighter. **Action for users:** if you
  intentionally ran unbounded loops longer than 50 iterations under the old
  default, set `max_iterations=None` (or your desired cap) explicitly.

## [0.3.0] — 2026-05-30

### Added
- **Opt-in anonymous funnel telemetry** (`loopgain.funnel`). A new,
  *separate* telemetry path from the per-loop product receiver
  (`loopgain.telemetry` / `LoopGain.send_telemetry`). It measures the
  project's adoption funnel — install → first `observe()` → recurring use —
  across the whole open-source userbase, so the maintainer can tell whether
  anyone is using the library. **Opt-in, default-decline:** nothing leaves
  the machine unless you explicitly opt in via `LOOPGAIN_TELEMETRY=1` or
  `loopgain telemetry --enable`. `LOOPGAIN_TELEMETRY=0` and `DO_NOT_TRACK=1`
  are honored as hard opt-outs; CI is auto-detected and declined silently.
  When enabled, payloads carry only anonymous counts/metadata — a
  locally-generated random instance id (not derived from your machine or
  identity), hour-bucketed timestamps, library/Python/OS versions, the
  framework adapter in use, and a coarse loop-outcome distribution. **Never
  sent:** prompts, outputs, error contents, keys, paths, or IPs. Delivery is
  batched, async (daemon thread + `atexit`), https-only, and fully
  fail-silent — a funnel error can never raise into your loop. Privacy
  contract is enforced by `tests/test_funnel.py`. See `TELEMETRY.md`.
- **`loopgain` command-line interface.** New console entry point.
  `loopgain telemetry --show | --enable | --disable | --reset` inspects and
  controls funnel telemetry; `loopgain version` prints the library version.
  Also runnable as `python -m loopgain`.

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
