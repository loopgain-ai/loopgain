"""Unit tests for the LoopGain decision engine.

Covers the five band transitions, the three canonical scenarios from the
spec (converging / oscillating / diverging), TARGET_MET short-circuit,
best-so-far buffer correctness, and ETA calibration.
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


def _decay(ab: float, e0: float = 100.0, n: int = 20) -> list[float]:
    """Generate a synthetic error trace with constant loop gain Aβ."""
    return [e0 * (ab**i) for i in range(n)]


# ----- Band classification (ThresholdBands.state_for) -----


def test_thresholds_classify_fast_converge():
    assert ThresholdBands().state_for(0.1) == FAST_CONVERGE
    assert ThresholdBands().state_for(0.29) == FAST_CONVERGE


def test_thresholds_classify_converging():
    assert ThresholdBands().state_for(0.3) == CONVERGING
    assert ThresholdBands().state_for(0.5) == CONVERGING
    assert ThresholdBands().state_for(0.84) == CONVERGING


def test_thresholds_classify_stalling():
    assert ThresholdBands().state_for(0.85) == STALLING
    assert ThresholdBands().state_for(0.9) == STALLING
    assert ThresholdBands().state_for(0.94) == STALLING


def test_thresholds_classify_oscillating():
    assert ThresholdBands().state_for(0.95) == OSCILLATING
    assert ThresholdBands().state_for(1.0) == OSCILLATING
    assert ThresholdBands().state_for(1.05) == OSCILLATING


def test_thresholds_classify_diverging():
    assert ThresholdBands().state_for(1.06) == DIVERGING
    assert ThresholdBands().state_for(1.5) == DIVERGING
    assert ThresholdBands().state_for(10.0) == DIVERGING


# ----- Three canonical scenarios from the spec -----


def test_canonical_converging_reaches_target():
    """Aβ ≈ 0.65: loop converges to target before max_iterations."""
    guard = LoopGain(target_error=0.5, max_iterations=20)
    errors = _decay(0.65, e0=100.0)
    for e in errors:
        if not guard.should_continue():
            break
        guard.observe(e)
    result = guard.result
    assert result.outcome == "converged"
    assert result.iterations_used < 20  # didn't hit cap
    assert result.best_error <= 0.5


def test_canonical_oscillating_terminates_with_best_so_far():
    """Aβ ≈ 1.0: loop never converges — should return best-so-far."""
    guard = LoopGain(max_iterations=20)
    # Constant errors (Aβ = 1.0 exactly)
    for i in range(20):
        if not guard.should_continue():
            break
        guard.observe(50.0, output=f"iter-{i}")
    result = guard.result
    assert result.outcome == "oscillating"
    assert result.iterations_used < 20  # terminated by stability detection
    # best_output is well-defined even with constant errors (first iter wins by argmin tiebreak)
    assert result.best_output is not None


def test_canonical_diverging_returns_pre_divergence_best():
    """Aβ ≈ 1.18: errors grow each iteration — best is an early iter."""
    guard = LoopGain(max_iterations=20)
    errors = _decay(1.18, e0=10.0)
    for i, e in enumerate(errors):
        if not guard.should_continue():
            break
        guard.observe(e, output=f"iter-{i}")
    result = guard.result
    assert result.outcome == "diverged"
    # Best output should be iter-0 (lowest error in a monotonically growing sequence).
    assert result.best_output == "iter-0"
    assert result.best_error == errors[0]


# ----- TARGET_MET short-circuit -----


def test_target_met_short_circuits():
    """Error below target stops the loop immediately, even if Aβ would say continue."""
    guard = LoopGain(target_error=0.5)
    guard.observe(10.0)
    state = guard.observe(0.4)  # below target
    assert state == TARGET_MET
    assert not guard.should_continue()
    assert guard.result.outcome == "converged"


def test_target_met_zero_is_disabled():
    """target_error=0 should NOT trigger TARGET_MET on a zero observation."""
    guard = LoopGain(target_error=0.0, max_iterations=5)
    guard.observe(10.0)
    state = guard.observe(0.0)
    # With target_error=0, zero error does not stop the loop early via TARGET_MET.
    # (It might trigger DIVERGING/etc via Aβ math, but not TARGET_MET.)
    assert state != TARGET_MET


# ----- Best-so-far buffer -----


def test_best_so_far_returns_minimum_index():
    guard = LoopGain(max_iterations=10)
    errors = [12.0, 4.0, 2.0, 0.8, 1.5, 3.0]
    for i, e in enumerate(errors):
        if not guard.should_continue():
            break
        guard.observe(e, output=f"out-{i}")
    result = guard.result
    assert result.best_index == 3  # 0.8 is the minimum
    assert result.best_output == "out-3"
    assert result.best_error == 0.8


def test_best_so_far_works_without_outputs():
    """If outputs aren't passed, best_output is None but best_index is still correct."""
    guard = LoopGain(max_iterations=10)
    for e in [10.0, 5.0, 2.0, 8.0]:
        if not guard.should_continue():
            break
        guard.observe(e)
    result = guard.result
    assert result.best_index == 2
    assert result.best_output is None
    assert result.best_error == 2.0


# ----- ETA prediction calibration -----


def test_eta_prediction_calibration():
    """Synthetic monotonic decay: predicted iterations matches actual ±1."""
    target = 0.1
    ab = 0.5
    errors = _decay(ab, e0=100.0)
    # Iteration index where error first drops below target.
    actual_n = next(i for i, e in enumerate(errors) if e < target)

    guard = LoopGain(target_error=target, max_iterations=100)
    # Feed a few errors so smoothed Aβ stabilizes.
    guard.observe(errors[0])
    guard.observe(errors[1])
    guard.observe(errors[2])
    # ETA at this point should predict ~(actual_n - 2) more iterations.
    eta = guard.eta
    assert eta is not None
    assert abs(eta - (actual_n - 2)) <= 1, f"eta={eta}, expected ~{actual_n - 2}"


def test_eta_none_when_not_converging():
    """ETA is None when Aβ_smooth >= 1 (non-converging)."""
    guard = LoopGain(target_error=0.5)
    for _ in range(3):
        guard.observe(10.0)  # constant errors -> Aβ = 1
        if not guard.should_continue():
            break
    assert guard.eta is None


def test_eta_none_when_target_is_zero():
    guard = LoopGain(target_error=0.0)
    guard.observe(100.0)
    guard.observe(50.0)
    assert guard.eta is None


# ----- observe() input coercion -----


def test_observe_accepts_number():
    guard = LoopGain()
    guard.observe(10.0)
    guard.observe(5.0)
    assert guard.state in (FAST_CONVERGE, CONVERGING)


def test_observe_accepts_int():
    guard = LoopGain()
    guard.observe(10)
    guard.observe(5)
    assert len(guard.result.error_history) == 2


def test_observe_accepts_sequence():
    guard = LoopGain()
    guard.observe(["e1", "e2", "e3"])  # magnitude = 3
    guard.observe(["e1", "e2"])  # magnitude = 2 → Aβ ≈ 0.67 → CONVERGING
    assert guard.state == CONVERGING


def test_observe_rejects_negative_number():
    guard = LoopGain()
    with pytest.raises(ValueError):
        guard.observe(-1.0)


def test_observe_rejects_nan():
    guard = LoopGain()
    with pytest.raises(ValueError):
        guard.observe(float("nan"))


def test_observe_rejects_unknown_type():
    guard = LoopGain()
    with pytest.raises(TypeError):
        guard.observe(object())


# ----- max_iterations safety cap -----


def test_max_iterations_triggers_terminal_state():
    guard = LoopGain(max_iterations=3)
    for _ in range(5):
        if not guard.should_continue():
            break
        guard.observe(10.0)
    # Constant errors will trigger OSCILLATING before MAX_ITERATIONS if window is large enough.
    # With max=3 and constant errors, we get OSCILLATING on iter 3 (Aβ=1.0 → OSCILLATING band).
    assert guard.state in (OSCILLATING, MAX_ITERATIONS)
    assert not guard.should_continue()


def test_max_iterations_with_converging_loop():
    """If max_iterations hits before convergence (and Aβ stays in a
    non-terminal band), the terminal state is MAX_ITERATIONS."""
    # Aβ = 0.5: clean CONVERGING band; cap at 2 forces MAX_ITERATIONS.
    guard = LoopGain(target_error=0.001, max_iterations=2)
    guard.observe(100.0)
    guard.observe(50.0)
    assert guard.state == MAX_ITERATIONS
    assert guard.result.outcome == "max_iterations"
    assert not guard.should_continue()


# ----- gain_margin -----


def test_gain_margin_greater_than_one_for_converging():
    guard = LoopGain(max_iterations=10)
    for e in _decay(0.5)[:5]:
        guard.observe(e)
    gm = guard.gain_margin
    assert gm is not None
    assert gm > 1.0


def test_gain_margin_none_before_observations():
    guard = LoopGain()
    assert guard.gain_margin is None


# ----- result before any observations -----


def test_result_not_started():
    guard = LoopGain()
    r = guard.result
    assert r.outcome == "not_started"
    assert r.iterations_used == 0
    assert r.best_index == -1


# ----- Constructor validation -----


def test_constructor_rejects_negative_target():
    with pytest.raises(ValueError):
        LoopGain(target_error=-0.1)


def test_constructor_rejects_zero_window():
    with pytest.raises(ValueError):
        LoopGain(smoothing_window=0)


def test_constructor_rejects_zero_max_iterations():
    with pytest.raises(ValueError):
        LoopGain(max_iterations=0)


# ----- Custom thresholds -----


def test_custom_thresholds_used():
    """Tighter STALLING boundary catches stalls earlier."""
    custom = ThresholdBands(
        fast_converge=0.2,
        converging=0.6,
        stalling=0.8,
        oscillating_upper=1.05,
    )
    assert custom.state_for(0.7) == STALLING  # would have been CONVERGING under defaults
    assert custom.state_for(0.5) == CONVERGING
    assert custom.state_for(0.1) == FAST_CONVERGE


# ----- savings_vs_fixed_cap -----


def test_savings_positive_when_converging_early():
    guard = LoopGain(target_error=0.5, assumed_fixed_cap=10)
    for e in [10.0, 1.0, 0.1]:
        if not guard.should_continue():
            break
        guard.observe(e)
    assert guard.result.savings_vs_fixed_cap > 0


def test_savings_zero_when_max_iterations_hit():
    guard = LoopGain(max_iterations=5, assumed_fixed_cap=10)
    for _ in range(10):
        if not guard.should_continue():
            break
        guard.observe(10.0)  # never converges
    result = guard.result
    if result.outcome == "max_iterations":
        assert result.savings_vs_fixed_cap == 0
