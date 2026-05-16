# Changelog

All notable changes to the `loopgain` library are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
