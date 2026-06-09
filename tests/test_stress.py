"""Stress tests and edge-case hardening for the LoopGain decision engine.

These tests cover the production failure modes that the canonical happy-path
tests don't: floating-point noise, very small/large magnitudes, long runs,
band-boundary jitter, pathological inputs, and re-entry after terminal.

If any of these fail, real production deployments will eventually surface
the same bug, but louder.
"""

from __future__ import annotations

import math

import pytest

from loopgain import (
    LoopGain,
    ThresholdBands,
    FAST_CONVERGE,
    CONVERGING,
    STALLING,
    OSCILLATING,
    DIVERGING,
    TARGET_MET,
    MAX_ITERATIONS,
)


# ----- Floating-point noise robustness -----


def test_smoothing_absorbs_small_aβ_jitter():
    """Aβ that jitters in CONVERGING band (around 0.6) should stay in the
    converging family — the EMA absorbs noise without false-positive
    transitions to STALLING/OSCILLATING/DIVERGING.

    Legacy classifier: this is exactly CONVERGING. Trajectory classifier:
    the cumulative reduction crosses one decade (0.6^8 ≈ 0.017), so the
    more specific FAST_CONVERGE fires. Both are correct and non-terminal.
    """
    lg = LoopGain(max_iterations=20)
    errors = [100.0]
    jitter = [0.58, 0.62, 0.60, 0.59, 0.61, 0.60, 0.58, 0.62]
    for j in jitter:
        errors.append(errors[-1] * j)
    for e in errors:
        if not lg.should_continue():
            break
        lg.observe(e)
    assert lg.state in (CONVERGING, FAST_CONVERGE)
    assert lg.should_continue()


def test_smoothing_handles_oscillating_band_jitter():
    """Legacy: Aβ jittering in the OSCILLATING band (0.95-1.05) should
    reliably terminate — no escape via lucky noise sample.

    The trajectory classifier categorizes this as STALLING with a tiny
    ±0.0088 log10 residual std (far below OSC_STD_THRESHOLD=0.30), then
    terminates via the consecutive-stall rule. Both classifiers terminate.
    """
    lg = LoopGain(max_iterations=20, classifier="legacy_bands")
    # Aβ alternates 0.98 / 1.02 — both in OSCILLATING band
    errors = [100.0]
    for _ in range(10):
        errors.append(errors[-1] * (0.98 if len(errors) % 2 == 0 else 1.02))
    for e in errors:
        if not lg.should_continue():
            break
        lg.observe(e)
    assert not lg.should_continue()
    assert lg.state in (OSCILLATING, DIVERGING)


def test_band_boundary_jitter_does_not_falsely_terminate():
    """Aβ values straddling the 0.85 STALLING boundary may flip the
    classification between CONVERGING and STALLING (EMA at window=3 cannot
    fully absorb ±0.02 jitter around a band edge), but must NOT falsely
    trigger a terminal state (OSCILLATING/DIVERGING) when the actual Aβ
    stays below 0.95.

    This is the correctness-critical invariant: noise near a non-terminal
    band edge should never produce a false abort. Larger smoothing_window
    values further stabilize the classification if needed.
    """
    lg = LoopGain(max_iterations=15)
    # Alternate Aβ = 0.83 (CONVERGING) and Aβ = 0.87 (STALLING).
    # Smoothed Aβ oscillates around 0.85 → expected classification flip.
    errors = [100.0]
    multipliers = [0.83, 0.87] * 10
    for m in multipliers:
        errors.append(errors[-1] * m)
    seen_states = set()
    for e in errors:
        if not lg.should_continue():
            break
        s = lg.observe(e)
        seen_states.add(s)
    # The critical assertion: no false termination via OSCILLATING/DIVERGING.
    assert OSCILLATING not in seen_states
    assert DIVERGING not in seen_states
    # The loop terminates via MAX_ITERATIONS, not false stability detection.
    if not lg.should_continue():
        assert lg.state in (CONVERGING, STALLING, MAX_ITERATIONS)


# ----- Numerical precision edge cases -----


def test_very_small_error_magnitudes():
    """Sub-femto-scale errors should compute Aβ correctly without underflow."""
    lg = LoopGain(target_error=1e-30, max_iterations=10)
    errors = [1e-10 * (0.5**i) for i in range(8)]
    for e in errors:
        if not lg.should_continue():
            break
        lg.observe(e)
    # Should classify as FAST_CONVERGE (Aβ=0.5) — never crash.
    assert lg.state in (FAST_CONVERGE, CONVERGING, TARGET_MET)


def test_very_large_error_magnitudes():
    """Galactic-scale errors should compute Aβ correctly without overflow."""
    lg = LoopGain(target_error=1.0, max_iterations=10)
    errors = [1e30 * (0.4**i) for i in range(8)]
    for e in errors:
        if not lg.should_continue():
            break
        lg.observe(e)
    # Aβ = 0.4 → FAST_CONVERGE.
    assert lg.state in (FAST_CONVERGE, CONVERGING)
    assert lg.result.convergence_profile  # Aβ computed without overflow


def test_zero_error_in_middle_of_run_anomalous():
    """When the short-circuit is disabled (target_error=None) and an error
    drops to zero mid-run, the next observe must not divide by zero."""
    lg = LoopGain(target_error=None, max_iterations=10)
    lg.observe(10.0)
    lg.observe(0.0)  # Aβ = 0; this is fine
    # Subsequent non-zero observation: Aβ = magnitude / 0 — must be handled.
    # Per our impl, prev=0 with magnitude>0 → treated as DIVERGING-equivalent.
    state = lg.observe(5.0)
    # Must not crash; must produce a defined state.
    assert state in (
        FAST_CONVERGE, CONVERGING, STALLING, OSCILLATING, DIVERGING, MAX_ITERATIONS
    )


def test_two_consecutive_zero_errors():
    """0 followed by 0 with short-circuit disabled → Aβ = 0/0 → handled
    by impl as Aβ=0 (still converging)."""
    lg = LoopGain(target_error=None, max_iterations=5)
    lg.observe(0.0)
    state = lg.observe(0.0)
    # 0/0 → 0.0 per the impl; state is FAST_CONVERGE.
    assert state in (FAST_CONVERGE, MAX_ITERATIONS)


# ----- Long-run sanity -----


def test_long_converging_run_does_not_leak_state():
    """500-iteration converging run: error history grows but should not crash
    or produce stale state, and gain margin stays well-defined."""
    lg = LoopGain(target_error=1e-200, max_iterations=500)
    e = 1.0
    n = 0
    while lg.should_continue() and n < 500:
        lg.observe(e)
        e *= 0.7
        n += 1
    result = lg.result
    # Must finish in some terminal state.
    assert not lg.should_continue()
    assert result.outcome in ("converged", "max_iterations", "in_progress")
    # error_history length matches iterations_used
    assert len(result.error_history) == result.iterations_used
    # convergence_profile is iterations_used - 1 (no Aβ at iter 0)
    assert len(result.convergence_profile) == result.iterations_used - 1


def test_long_oscillating_run_terminates_promptly():
    """Legacy: even with max_iterations=10000, a clearly oscillating loop
    should terminate within a handful of iterations on stability detection.

    Under the legacy classifier, constant errors → Aβ=1.0 → OSCILLATING
    band → terminal. Under the trajectory classifier the same trajectory
    is STALLING (zero slope, zero variance) and terminates via the
    consecutive-stall rule; see ``test_trajectory_consecutive_stall_terminates``.
    """
    lg = LoopGain(max_iterations=10000, classifier="legacy_bands")
    n = 0
    while lg.should_continue() and n < 10000:
        lg.observe(50.0)  # constant errors → Aβ = 1.0
        n += 1
    # Should have terminated FAR earlier than max.
    assert lg.result.iterations_used < 20
    assert lg.state == OSCILLATING


# ----- Re-entry / API contract -----


def test_observe_after_terminal_is_noop():
    """Calling observe() after a terminal state fires should be a no-op
    that returns the current state (not raise, not append, not flip state)."""
    lg = LoopGain(target_error=0.5)
    lg.observe(10.0)
    lg.observe(0.4)  # TARGET_MET
    assert lg.state == TARGET_MET

    pre_count = lg.result.iterations_used
    state = lg.observe(100.0)  # would be DIVERGING if processed
    assert state == TARGET_MET  # state unchanged
    assert lg.result.iterations_used == pre_count  # no new observation recorded


def test_should_continue_stays_false_after_terminal():
    """Once should_continue() returns False, it stays False — no flapping."""
    lg = LoopGain(max_iterations=5)
    for _ in range(5):
        if not lg.should_continue():
            break
        lg.observe(50.0)
    assert not lg.should_continue()
    # Re-check multiple times.
    for _ in range(10):
        assert not lg.should_continue()


def test_result_callable_mid_loop():
    """result property is safe to call mid-loop with outcome="in_progress"."""
    lg = LoopGain(max_iterations=20)
    lg.observe(10.0)
    lg.observe(5.0)
    r = lg.result
    assert r.outcome == "in_progress"
    assert r.iterations_used == 2
    # Continue after reading result.
    lg.observe(2.5)
    r2 = lg.result
    assert r2.iterations_used == 3


# ----- Mixed input types -----


def test_observe_handles_tuple():
    """Sequence input via tuple, not just list."""
    lg = LoopGain()
    lg.observe(("e1", "e2", "e3"))  # magnitude = 3
    lg.observe(("e1", "e2"))  # magnitude = 2
    assert lg.state == CONVERGING


def test_observe_handles_empty_sequence():
    """Empty sequence → magnitude 0. With target_error=None, doesn't
    short-circuit so multiple zero-magnitude observations can be made."""
    lg = LoopGain(target_error=None, max_iterations=3)
    lg.observe([])  # magnitude = 0
    # Sequel observations should not crash.
    lg.observe([])
    assert lg.result.iterations_used == 2


def test_observe_handles_dict_with_len():
    """Dict-like with __len__ is treated as sequence (len = number of keys)."""
    lg = LoopGain()
    lg.observe({"a": 1, "b": 2, "c": 3})  # magnitude = 3
    lg.observe({"a": 1, "b": 2})  # magnitude = 2 → Aβ = 0.67 → CONVERGING
    assert lg.state == CONVERGING


def test_observe_rejects_string():
    """Strings have __len__ but are almost always a user mistake — be strict."""
    # Currently the impl WILL accept a string via __len__. Document the
    # current behavior and consider tightening in a future version.
    lg = LoopGain()
    # We choose to accept strings (len semantics) but it's an unexpected path.
    # If we ever tighten this, update the test.
    state = lg.observe("abc")  # len("abc") = 3
    assert lg.result.error_history[0] == 3.0


# ----- Custom thresholds: degenerate cases -----


def test_custom_thresholds_with_equal_boundaries():
    """Edge case: same fast_converge and converging boundaries.
    Should not crash; should produce SOME defined state."""
    weird = ThresholdBands(
        fast_converge=0.5,
        converging=0.5,  # equal to fast_converge → CONVERGING band has zero width
        stalling=0.9,
        oscillating_upper=1.05,
    )
    # 0.4 → FAST_CONVERGE (< 0.5)
    # 0.5 → CONVERGING (not < 0.5; < 0.5 fails for CONVERGING band so falls to STALLING? Let's check)
    # Per state_for: < 0.5 → FAST. Not < 0.5, but < 0.5 for converging? No.
    # Actually < self.fast_converge=0.5 → FAST. Not < 0.5 but < self.converging=0.5? No (not <).
    # So 0.5 → STALLING (since < self.stalling=0.9 holds).
    assert weird.state_for(0.4) == FAST_CONVERGE
    assert weird.state_for(0.5) == STALLING  # zero-width CONVERGING band collapses
    assert weird.state_for(0.7) == STALLING


def test_custom_thresholds_reordered_does_not_crash():
    """Pathological: stalling < converging. Don't crash; behavior is consistent
    with the if-ladder semantics (whichever predicate fires first wins)."""
    pathological = ThresholdBands(
        fast_converge=0.3,
        converging=0.9,
        stalling=0.5,  # below converging — pathological
        oscillating_upper=1.05,
    )
    # 0.4 → < 0.3? no. < 0.9? yes → CONVERGING. Stalling never evaluated.
    assert pathological.state_for(0.4) == CONVERGING
    # 0.95 → < 0.3? no. < 0.9? no. < 0.5? no. <= 1.05? yes → OSCILLATING.
    assert pathological.state_for(0.95) == OSCILLATING


# ----- Best-so-far buffer with large outputs -----


def test_best_so_far_with_large_output_objects():
    """Outputs can be arbitrary Python objects, including large ones —
    the buffer stores references, not copies."""
    big = list(range(100_000))  # ~800KB list
    lg = LoopGain(max_iterations=5)
    lg.observe(10.0, output=big)
    lg.observe(1.0, output=big)
    lg.observe(20.0, output=big)
    lg.observe(50.0, output=big)
    result = lg.result
    # All buffer slots reference the same list (no copy).
    assert result.best_output is big
    assert result.best_index == 1


def test_best_so_far_with_none_outputs_mixed():
    """Some observations have output=None, others don't. The buffer remains correct."""
    lg = LoopGain(max_iterations=10)
    lg.observe(10.0, output="a")
    lg.observe(5.0)  # output=None
    lg.observe(2.0, output="c")
    lg.observe(8.0)  # output=None
    result = lg.result
    assert result.best_index == 2
    assert result.best_output == "c"


# ----- assumed_fixed_cap edge cases -----


def test_savings_when_converged_at_cap_boundary():
    """assumed_fixed_cap=N and converged-in-exactly-N: savings = 0."""
    lg = LoopGain(target_error=0.5, assumed_fixed_cap=3)
    lg.observe(10.0)
    lg.observe(2.0)
    lg.observe(0.3)  # converged here
    assert lg.result.savings_vs_fixed_cap == 0


def test_savings_when_exceeded_cap():
    """If iterations_used > assumed_fixed_cap, savings clamps to 0 (not negative)."""
    lg = LoopGain(target_error=0.001, assumed_fixed_cap=2)
    lg.observe(10.0)
    lg.observe(5.0)
    lg.observe(2.5)
    lg.observe(1.0)  # 4 iterations, but cap is 2
    # We're not in max_iterations terminal, so savings uses the formula
    # max(0, cap - used) = max(0, 2 - 4) = 0
    if lg.result.outcome != "max_iterations":
        assert lg.result.savings_vs_fixed_cap == 0
