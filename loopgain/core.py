"""LoopGain — Barkhausen stability monitor for AI agent loops.

The product layer of the Barkhausen stability criterion (1921) applied to
any iterative AI loop with a measurable error signal: verify-revise (GVR)
patterns, refinement passes, tool-use retry chains, RAG with self-correction,
code-gen with linter feedback, multi-step reasoning loops, and custom
feedback systems. Replaces the universal max_iterations hack with a
real-time loop-gain monitor that classifies the loop into one of five
named states and decides whether to continue, stop, or roll back.

The math is foundational EE control theory. The product layer is the
threshold bands, the best-so-far buffer, and the clean Python API.

License: Apache-2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from loopgain import funnel
from loopgain.classifier import (
    TrajectoryThresholds,
    classify_trajectory,
    extract_features,
)


# Canonical threshold bands (Aβ_smooth axis).
# Used by the legacy single-feature classifier (see ThresholdBands.state_for).
# The v0.2 default classifier is the multi-feature trajectory classifier in
# ``loopgain.classifier``; these bands remain for callers that explicitly opt
# into the legacy behavior via ``classifier='legacy_bands'``.
DEFAULT_FAST_CONVERGE = 0.3
DEFAULT_CONVERGING = 0.85
DEFAULT_STALLING = 0.95
DEFAULT_OSCILLATING_UPPER = 1.05


# Bounded-by-default safety backstop. The loop should normally terminate on a
# stability verdict (target met / oscillating / diverging / stalled) long
# before this; it exists only so the library can never run truly unbounded if
# a loop never converges and never stalls (e.g. infinitesimal-but-real progress
# with target_error=None). Generous relative to typical loop lengths (the
# bench capped at 20). Pass max_iterations=None to opt into a fully unbounded
# loop, or a smaller integer to cap tighter.
DEFAULT_MAX_ITERATIONS = 50


# State names. Exported for use in switch/case in user code.
INIT = "INIT"
FAST_CONVERGE = "FAST_CONVERGE"
CONVERGING = "CONVERGING"
STALLING = "STALLING"
OSCILLATING = "OSCILLATING"
DIVERGING = "DIVERGING"
TARGET_MET = "TARGET_MET"
MAX_ITERATIONS = "MAX_ITERATIONS"


@dataclass(frozen=True)
class ThresholdBands:
    """Aβ_smooth threshold bands for the decision engine.

    Each band partitions the smoothed loop-gain axis. The defaults are the
    canonical v0.1 bands (0.3 / 0.85 / 0.95 / 1.05). Custom bands can be
    passed for per-domain tuning once you have production traces.
    """

    fast_converge: float = DEFAULT_FAST_CONVERGE
    converging: float = DEFAULT_CONVERGING
    stalling: float = DEFAULT_STALLING
    oscillating_upper: float = DEFAULT_OSCILLATING_UPPER

    def state_for(self, ab_smooth: float) -> str:
        """Classify a smoothed Aβ value into one of the five bands."""
        if ab_smooth < self.fast_converge:
            return FAST_CONVERGE
        if ab_smooth < self.converging:
            return CONVERGING
        if ab_smooth < self.stalling:
            return STALLING
        if ab_smooth <= self.oscillating_upper:
            return OSCILLATING
        return DIVERGING


@dataclass
class LoopGainResult:
    """Terminal result of a LoopGain-monitored loop.

    Returned by ``LoopGain.result``. Safe to read at any time, including
    mid-loop (in which case ``outcome`` is ``"in_progress"``).
    """

    outcome: str
    """One of "converged", "oscillating", "diverged", "max_iterations",
    "in_progress", "not_started"."""

    iterations_used: int
    """Number of ``observe()`` calls made."""

    best_index: int
    """Index of the iteration with the lowest observed error.
    ``-1`` if no observations have been made."""

    best_output: Any = None
    """Output associated with ``best_index``, if outputs were passed to
    ``observe()``. ``None`` otherwise."""

    best_error: float = 0.0
    """The lowest observed error magnitude."""

    convergence_profile: list[float] = field(default_factory=list)
    """Smoothed Aβ values at each iteration. Length is
    ``iterations_used - 1`` (no Aβ for the first observation)."""

    error_history: list[float] = field(default_factory=list)
    """All observed error magnitudes, in order."""

    savings_vs_fixed_cap: Optional[int] = None
    """Iterations saved versus the assumed fixed cap (default 10).
    Zero if the loop hit ``max_iterations``; otherwise non-negative."""


class LoopGain:
    """Barkhausen stability monitor for AI agent loops.

    Wraps any iterative loop with a measurable error signal and decides in
    real time whether to continue, stop, or roll back. Works for
    verify-revise (GVR) loops, refinement passes, tool-use retry chains,
    RAG with self-correction, code-gen with linter feedback, multi-step
    reasoning loops, and any custom iterative process where you can
    produce a number that should drop toward zero. Replaces the universal
    ``max_iterations=N`` hack with a control-theoretic stability monitor
    based on the Barkhausen criterion.

    Example:

        >>> from loopgain import LoopGain
        >>> lg = LoopGain(target_error=0.1)
        >>> while lg.should_continue():
        ...     errors = verifier.verify(output)
        ...     lg.observe(errors, output=output)
        ...     output = reviser.revise(output, errors)
        >>> result = lg.result
        >>> result.outcome          # "converged" | "oscillating" | ...
        >>> result.best_output      # lowest-error iteration's output

    Args:
        target_error: Stop when an observed error drops at or below this.
            Default ``0.0`` short-circuits on exactly zero error — the
            natural completion signal for most verifiers (no failing
            tests, no validation errors, etc.). Pass ``None`` to disable
            the short-circuit entirely and rely only on stability
            detection and ``max_iterations``.
        max_iterations: Hard safety backstop. Default
            ``DEFAULT_MAX_ITERATIONS`` (50) so the loop can never run
            unbounded; normally a stability verdict terminates it long
            before this. Pass ``None`` to opt into a fully unbounded loop,
            or a smaller integer to cap tighter.
        thresholds: Custom ``ThresholdBands`` (legacy single-feature
            classifier only). Default is the canonical 0.3 / 0.85 / 0.95 /
            1.05. Ignored when ``classifier='trajectory'``.
        trajectory_thresholds: Custom ``TrajectoryThresholds`` for the v0.2
            multi-feature classifier. Default is the pre-registered set
            in ``PROTOCOL_v2_classifier.md``. Ignored when
            ``classifier='legacy_bands'``.
        classifier: ``'trajectory'`` (default) uses the v0.2 multi-feature
            trajectory classifier. ``'legacy_bands'`` uses the v0.1
            single-feature Aβ-band classifier (kept for callers that
            empirically tuned ``ThresholdBands`` against a specific
            workload).
        smoothing_window: EMA window for ``Aβ_smooth``. Default 3. The Aβ
            series is computed and stored regardless of which classifier is
            in use — telemetry payloads always include the convergence
            profile.
        assumed_fixed_cap: Used to compute ``savings_vs_fixed_cap``.
            Default 10 (a generous default agent iteration cap).
        stall_terminate_count: Number of *consecutive* STALLING readings that
            terminate the loop (trajectory classifier only). Default ``2``,
            which is tuned for inner / per-generation loops where a brief
            plateau is a reliable stop signal. Session-scale / outer loops
            (e.g. Ralph-style runs where each iteration is a whole agent
            session) should raise this — a single regression-then-recovery
            session shows up as a transient stall, and the impatient default-2
            kill stops too early; the 2026-06-11 outer-loop study showed the
            consensus best value is around ``5`` (raising the count retained
            every true catch while roughly halving false stops). The exact
            session-scale default is not yet statistically pinned (a separate
            ~1000-run study is pending), so the library ships the conservative
            inner-loop default and exposes this knob. Must be ``>= 1``; this is
            distinct from ``TrajectoryThresholds.stall_patience``, which
            governs STALLING *onset* (how long a plateau must persist before
            the loop is *labelled* STALLING), not how many consecutive
            STALLING labels terminate it. Ignored under ``classifier=
            'legacy_bands'`` (legacy bands keep their non-terminal-STALLING
            contract).
    """

    def __init__(
        self,
        target_error: Optional[float] = 0.0,
        max_iterations: Optional[int] = DEFAULT_MAX_ITERATIONS,
        thresholds: Optional[ThresholdBands] = None,
        trajectory_thresholds: Optional[TrajectoryThresholds] = None,
        classifier: str = "trajectory",
        smoothing_window: int = 3,
        assumed_fixed_cap: int = 10,
        stall_terminate_count: int = 2,
    ) -> None:
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if target_error is not None and target_error < 0:
            raise ValueError("target_error must be non-negative or None")
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be >= 1 or None")
        if classifier not in ("trajectory", "legacy_bands"):
            raise ValueError(
                "classifier must be 'trajectory' or 'legacy_bands'; got "
                + repr(classifier)
            )
        if stall_terminate_count < 1:
            raise ValueError("stall_terminate_count must be >= 1")

        self.target_error: Optional[float] = (
            float(target_error) if target_error is not None else None
        )
        self.max_iterations = max_iterations
        self.thresholds = thresholds or ThresholdBands()
        self.trajectory_thresholds = trajectory_thresholds or TrajectoryThresholds()
        self.classifier_kind = classifier
        self.smoothing_window = smoothing_window
        self.assumed_fixed_cap = assumed_fixed_cap
        self.stall_terminate_count = stall_terminate_count

        self._error_history: list[float] = []
        self._gain_history: list[float] = []
        self._smoothed_history: list[float] = []
        self._outputs: list[Any] = []
        self._state: str = INIT
        self._state_history: list[str] = []
        self._terminal: bool = False

        # Opt-in anonymous funnel telemetry (see loopgain.funnel). No-op
        # unless the user has explicitly opted in; fully fail-silent and
        # distinct from the product receiver in loopgain.telemetry.
        funnel.on_init()

    # ----- Public observation API -----

    def observe(self, errors: Any, output: Any = None) -> str:
        """Record this iteration's errors and (optional) output.

        Call once per iteration after running your verifier. Returns the
        post-observation state so callers can branch on transitions.

        Args:
            errors: Error signal for this iteration. Accepts:
                - A number (int/float): used directly as the error magnitude.
                - A sequence (list/tuple/etc with ``__len__``):
                  ``len(errors)`` is the magnitude.
            output: Optional output produced this iteration. If provided,
                stored in the best-so-far buffer so ``result.best_output``
                returns the output associated with the lowest error.

        Returns:
            Current state name.
        """
        if self._terminal:
            return self._state

        magnitude = self._coerce_error(errors)
        self._error_history.append(magnitude)
        self._outputs.append(output)

        # Funnel telemetry: the first observe() ever made is the activation
        # signal. No-op unless opted in; deduped to once-per-install.
        if len(self._error_history) == 1:
            funnel.on_first_observe()

        # TARGET_MET short-circuit takes precedence over band classification.
        # target_error=None disables the short-circuit entirely; any non-None
        # value (including 0.0) fires TARGET_MET when magnitude <= target.
        if self.target_error is not None and magnitude <= self.target_error:
            self._state = TARGET_MET
            self._terminal = True
            funnel.note_outcome(self._state)
            return self._state

        # Compute Aβ if we have a prior observation. The smoothed Aβ series
        # is always maintained — telemetry payloads include it regardless of
        # which classifier is selected, and the dashboard relies on it for
        # per-iteration coloring on the trajectory chart.
        if len(self._error_history) >= 2:
            prev = self._error_history[-2]
            if prev > 0:
                ab = magnitude / prev
            elif magnitude == 0:
                ab = 0.0
            else:
                # Previous was zero but we didn't hit target — anomalous.
                # Treat as a large finite gain to surface as DIVERGING.
                ab = self.thresholds.oscillating_upper + 1.0
            self._gain_history.append(ab)
            self._smoothed_history.append(self._compute_smoothed(ab))

            if self.classifier_kind == "legacy_bands":
                # v0.1: single-feature Aβ_smooth band classification.
                self._state = self.thresholds.state_for(
                    self._smoothed_history[-1]
                )
            else:
                # v0.2: trajectory classifier. See PROTOCOL_v2_classifier.md.
                self._state = classify_trajectory(
                    self._error_history,
                    target_error=None,  # target_error short-circuit handled above
                    thresholds=self.trajectory_thresholds,
                )

            if self._state in (OSCILLATING, DIVERGING):
                self._terminal = True

            # v0.2 trajectory classifier: STALLING terminates after
            # ``stall_terminate_count`` *consecutive* stall readings (matches
            # the v2 protocol's "2+ consecutive stall readings" rule by
            # default — `component-algebra-v2-protocol-final-4.md` §3.3,
            # "Stalling → Return best-so-far"). The count is configurable so
            # session-scale / outer loops can be more patient than the
            # inner-loop default of 2 (see ``stall_terminate_count``). The
            # current reading is STALLING; the prior ``count - 1`` readings in
            # ``_state_history`` must also all be STALLING. Indexing by an
            # explicit offset (not a negative slice) keeps ``count == 1``
            # correct — ``[-0:]`` would wrongly select the whole history.
            # Legacy bands keep their original non-terminal-STALLING contract.
            if self.classifier_kind == "trajectory" and self._state == STALLING:
                prior_needed = self.stall_terminate_count - 1
                recent = (
                    self._state_history[len(self._state_history) - prior_needed:]
                    if prior_needed > 0
                    else []
                )
                if len(self._state_history) >= prior_needed and all(
                    s == STALLING for s in recent
                ):
                    self._terminal = True

            self._state_history.append(self._state)
        else:
            # First observation: no Aβ yet. Conservative default state.
            self._state = FAST_CONVERGE

        # Hard max_iterations cap (only if not already terminal).
        if (
            self.max_iterations is not None
            and len(self._error_history) >= self.max_iterations
            and not self._terminal
        ):
            self._state = MAX_ITERATIONS
            self._terminal = True

        # Funnel telemetry: if this observation drove the loop terminal
        # (oscillating / diverging / stalled / max-iterations), count the
        # coarse outcome. The TARGET_MET case is handled at its early return.
        if self._terminal:
            funnel.note_outcome(self._state)

        return self._state

    def should_continue(self) -> bool:
        """Whether the loop should run another iteration.

        Returns ``True`` until ``observe()`` detects a terminal state
        (target met, oscillating, diverging, or max iterations reached).
        """
        return not self._terminal

    # ----- Computed properties -----

    @property
    def state(self) -> str:
        """Current state name."""
        return self._state

    @property
    def result(self) -> LoopGainResult:
        """Construct the terminal result. Safe to call any time."""
        if not self._error_history:
            return LoopGainResult(
                outcome="not_started",
                iterations_used=0,
                best_index=-1,
            )

        if self._state == TARGET_MET:
            outcome = "converged"
        elif self._state == OSCILLATING:
            outcome = "oscillating"
        elif self._state == DIVERGING:
            outcome = "diverged"
        elif self._state == MAX_ITERATIONS:
            outcome = "max_iterations"
        elif self._state == STALLING and self._terminal:
            # v0.2 trajectory classifier marks STALLING terminal after
            # ``stall_terminate_count`` consecutive stall readings (default 2;
            # v2 protocol §3.3, "Return best-so-far"). Surfaced as the
            # "stalled" outcome — distinct
            # from "oscillating" so callers can route on "stuck but not
            # flapping" vs. "actively unstable." Dashboard's bandFromEvent
            # maps "stalled" → STALLING band.
            outcome = "stalled"
        else:
            outcome = "in_progress"

        best_index = self._error_history.index(min(self._error_history))
        best_error = self._error_history[best_index]
        best_output = self._outputs[best_index] if best_index < len(self._outputs) else None

        if outcome != "max_iterations":
            savings = max(0, self.assumed_fixed_cap - len(self._error_history))
        else:
            savings = 0

        return LoopGainResult(
            outcome=outcome,
            iterations_used=len(self._error_history),
            best_index=best_index,
            best_output=best_output,
            best_error=best_error,
            convergence_profile=list(self._smoothed_history),
            error_history=list(self._error_history),
            savings_vs_fixed_cap=savings,
        )

    # ----- Internal helpers -----

    def _coerce_error(self, errors: Any) -> float:
        if isinstance(errors, bool):
            # Coerce True/False to 1/0 explicitly to avoid surprising int promotion.
            return float(int(errors))
        if isinstance(errors, (int, float)):
            if errors < 0:
                raise ValueError("error magnitude must be non-negative")
            if math.isnan(errors) or math.isinf(errors):
                raise ValueError("error magnitude must be finite")
            return float(errors)
        if hasattr(errors, "__len__"):
            return float(len(errors))
        raise TypeError(
            "observe() expected a number or sequence; got "
            + type(errors).__name__
        )

    def _compute_smoothed(self, latest_ab: float) -> float:
        """EMA over the configured window."""
        if not self._smoothed_history:
            return latest_ab
        alpha = 2.0 / (self.smoothing_window + 1)
        prior = self._smoothed_history[-1]
        return alpha * latest_ab + (1 - alpha) * prior

    # ----- Telemetry (opt-in) -----

    def send_telemetry(
        self,
        endpoint: str,
        token: str,
        workload_id: Optional[str] = None,
        timeout: float = 2.0,
        allow_insecure: bool = False,
        framework: Optional[str] = None,
        loop_type: Optional[str] = None,
        team: Optional[str] = None,
        include_per_iteration: bool = True,
        retries: int = 2,
        retry_backoff: float = 0.25,
        actual_dollars_spent: Optional[float] = None,
        actual_dollars_saved: Optional[float] = None,
    ) -> bool:
        """Send anonymized telemetry to a receiver endpoint.

        Opt-in. Call once after the loop terminates. Sends only structural
        statistics — Aβ values, error magnitudes, state transitions, gain
        margin, rollback flag, library version, and optional opaque labels.
        Never sends prompts, completions, error contents, or customer
        identity beyond the bearer token.

        Best-effort: errors are swallowed; never raises. Safe to call from
        within an exception handler or finally block.

        Args:
            endpoint: Telemetry receiver URL. Must use ``https://``;
                ``http://`` is rejected unless ``allow_insecure`` is ``True``.
            token: Bearer token issued by the receiver (rotatable).
            workload_id: Optional opaque label that groups related loops in
                the dashboard. Never used to identify the customer.
            timeout: Per-request timeout in seconds. Default 2.0.
            allow_insecure: If ``True``, permit ``http://`` endpoints (for
                local development). Default ``False``.
            framework: Optional classification — agent framework name
                (``"langgraph"``, ``"crewai"``, etc.). Adapters auto-stamp.
            loop_type: Optional classification — loop pattern name
                (``"verify_revise"``, ``"rag_refine"``, etc.).
            team: Optional classification — team or environment label.
            include_per_iteration: If ``True`` (default), include the
                per-iteration Aβ + error trajectories (capped) so the
                dashboard's Loop Detail scrubber works. Set ``False`` to
                send only aggregate summary stats.
            retries: Additional attempts if a send fails *transiently*
                (timeout, connection error, 5xx/429). Default 2 (up to 3
                attempts). Set to 0 for single-shot. Deterministic failures
                (bad token, etc.) are never retried.
            retry_backoff: Base seconds between attempts; the nth retry waits
                ``retry_backoff * n``. Default 0.25.
            actual_dollars_spent: Optional real measured $ cost of this
                trial, when the caller has actual per-run cost data (e.g.
                summed token usage x list price). Omit to let the
                dashboard fall back to its iter-count x $/iter
                extrapolation.
            actual_dollars_saved: Optional real measured $ delta vs. a
                paired baseline run, when the caller has one. Same
                fallback semantics as ``actual_dollars_spent``.

        Returns:
            ``True`` on 2xx response, ``False`` otherwise.

        Example:
            >>> lg = LoopGain(target_error=0.1)
            >>> while lg.should_continue():
            ...     lg.observe(verifier.verify(output))
            ...     output = reviser.revise(output)
            >>> lg.send_telemetry(
            ...     endpoint="https://telemetry.loopgain.ai/v1/aggregate",
            ...     token="your-token-here",
            ...     workload_id="my-rag-pipeline",
            ...     framework="langgraph",
            ...     loop_type="verify_revise",
            ... )
        """
        from loopgain.telemetry import build_payload, send_payload

        payload = build_payload(
            self,
            workload_id=workload_id,
            framework=framework,
            loop_type=loop_type,
            team=team,
            include_per_iteration=include_per_iteration,
            actual_dollars_spent=actual_dollars_spent,
            actual_dollars_saved=actual_dollars_saved,
        )
        return send_payload(
            endpoint,
            token,
            payload,
            timeout=timeout,
            allow_insecure=allow_insecure,
            retries=retries,
            retry_backoff=retry_backoff,
        )
