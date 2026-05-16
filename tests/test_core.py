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
    lg = LoopGain(target_error=0.5, max_iterations=20)
    errors = _decay(0.65, e0=100.0)
    for e in errors:
        if not lg.should_continue():
            break
        lg.observe(e)
    result = lg.result
    assert result.outcome == "converged"
    assert result.iterations_used < 20  # didn't hit cap
    assert result.best_error <= 0.5


def test_canonical_oscillating_terminates_with_best_so_far():
    """Aβ ≈ 1.0: loop never converges — should return best-so-far."""
    lg = LoopGain(max_iterations=20)
    # Constant errors (Aβ = 1.0 exactly)
    for i in range(20):
        if not lg.should_continue():
            break
        lg.observe(50.0, output=f"iter-{i}")
    result = lg.result
    assert result.outcome == "oscillating"
    assert result.iterations_used < 20  # terminated by stability detection
    # best_output is well-defined even with constant errors (first iter wins by argmin tiebreak)
    assert result.best_output is not None


def test_canonical_diverging_returns_pre_divergence_best():
    """Aβ ≈ 1.18: errors grow each iteration — best is an early iter."""
    lg = LoopGain(max_iterations=20)
    errors = _decay(1.18, e0=10.0)
    for i, e in enumerate(errors):
        if not lg.should_continue():
            break
        lg.observe(e, output=f"iter-{i}")
    result = lg.result
    assert result.outcome == "diverged"
    # Best output should be iter-0 (lowest error in a monotonically growing sequence).
    assert result.best_output == "iter-0"
    assert result.best_error == errors[0]


# ----- TARGET_MET short-circuit -----


def test_target_met_short_circuits():
    """Error below target stops the loop immediately, even if Aβ would say continue."""
    lg = LoopGain(target_error=0.5)
    lg.observe(10.0)
    state = lg.observe(0.4)  # below target
    assert state == TARGET_MET
    assert not lg.should_continue()
    assert lg.result.outcome == "converged"


def test_target_error_zero_fires_target_met_on_exact_zero():
    """Default target_error=0.0 short-circuits when error hits exactly zero —
    the natural completion signal for verifier-driven loops (no failing
    tests, no validation errors, etc.)."""
    lg = LoopGain(max_iterations=5)  # default target_error=0.0
    lg.observe(10.0)
    state = lg.observe(0.0)
    assert state == TARGET_MET
    assert not lg.should_continue()


def test_target_error_none_disables_short_circuit():
    """Passing target_error=None disables the short-circuit entirely;
    only stability detection and max_iterations terminate the loop."""
    lg = LoopGain(target_error=None, max_iterations=5)
    lg.observe(10.0)
    state = lg.observe(0.0)
    # Zero observation does NOT trigger TARGET_MET with target_error=None.
    assert state != TARGET_MET


# ----- Best-so-far buffer -----


def test_best_so_far_returns_minimum_index():
    lg = LoopGain(max_iterations=10)
    errors = [12.0, 4.0, 2.0, 0.8, 1.5, 3.0]
    for i, e in enumerate(errors):
        if not lg.should_continue():
            break
        lg.observe(e, output=f"out-{i}")
    result = lg.result
    assert result.best_index == 3  # 0.8 is the minimum
    assert result.best_output == "out-3"
    assert result.best_error == 0.8


def test_best_so_far_works_without_outputs():
    """If outputs aren't passed, best_output is None but best_index is still correct."""
    lg = LoopGain(max_iterations=10)
    for e in [10.0, 5.0, 2.0, 8.0]:
        if not lg.should_continue():
            break
        lg.observe(e)
    result = lg.result
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

    lg = LoopGain(target_error=target, max_iterations=100)
    # Feed a few errors so smoothed Aβ stabilizes.
    lg.observe(errors[0])
    lg.observe(errors[1])
    lg.observe(errors[2])
    # ETA at this point should predict ~(actual_n - 2) more iterations.
    eta = lg.eta
    assert eta is not None
    assert abs(eta - (actual_n - 2)) <= 1, f"eta={eta}, expected ~{actual_n - 2}"


def test_eta_none_when_not_converging():
    """ETA is None when Aβ_smooth >= 1 (non-converging)."""
    lg = LoopGain(target_error=0.5)
    for _ in range(3):
        lg.observe(10.0)  # constant errors -> Aβ = 1
        if not lg.should_continue():
            break
    assert lg.eta is None


def test_eta_none_when_target_is_zero():
    lg = LoopGain(target_error=0.0)
    lg.observe(100.0)
    lg.observe(50.0)
    assert lg.eta is None


# ----- First-eta snapshot (for ETA Accuracy dashboard panel) -----


def test_first_eta_snapshot_captured_during_converging_run():
    """Result carries the first non-None eta and the iter it was made at."""
    target = 0.1
    errors = _decay(0.5, e0=100.0)
    lg = LoopGain(target_error=target, max_iterations=100)
    for e in errors:
        lg.observe(e)
        if not lg.should_continue():
            break

    result = lg.result
    # First eta becomes computable on the 2nd observation (smoothed_history
    # exists, target > 0, current > target, Aβ_smooth < 1).
    assert result.first_eta_at_iteration == 2
    assert result.first_eta_prediction is not None
    assert result.first_eta_prediction > 0
    # Predicted total iterations should be within ±2 of actual (the prediction
    # is made early, before smoothing has fully settled).
    predicted_total = result.first_eta_at_iteration + result.first_eta_prediction
    assert abs(predicted_total - result.iterations_used) <= 2


def test_first_eta_snapshot_none_when_target_is_zero():
    """No prediction is captured when target_error=0 (eta is always None)."""
    lg = LoopGain(target_error=0.0, max_iterations=5)
    for _ in range(5):
        lg.observe(10.0)
    result = lg.result
    assert result.first_eta_prediction is None
    assert result.first_eta_at_iteration is None


def test_first_eta_snapshot_none_when_loop_never_converges():
    """Oscillating loop (Aβ ≈ 1) never produces a positive eta."""
    lg = LoopGain(target_error=0.5)
    for _ in range(5):
        lg.observe(10.0)
        if not lg.should_continue():
            break
    result = lg.result
    assert result.first_eta_prediction is None
    assert result.first_eta_at_iteration is None


def test_first_eta_snapshot_is_idempotent():
    """Subsequent observations don't overwrite the first prediction."""
    target = 0.1
    errors = _decay(0.5, e0=100.0)
    lg = LoopGain(target_error=target, max_iterations=100)
    lg.observe(errors[0])
    lg.observe(errors[1])
    first = lg._first_eta_prediction
    first_iter = lg._first_eta_at_iteration
    assert first is not None
    # Run a few more iterations; the snapshot should not change.
    for e in errors[2:6]:
        lg.observe(e)
    assert lg._first_eta_prediction == first
    assert lg._first_eta_at_iteration == first_iter


# ----- observe() input coercion -----


def test_observe_accepts_number():
    lg = LoopGain()
    lg.observe(10.0)
    lg.observe(5.0)
    assert lg.state in (FAST_CONVERGE, CONVERGING)


def test_observe_accepts_int():
    lg = LoopGain()
    lg.observe(10)
    lg.observe(5)
    assert len(lg.result.error_history) == 2


def test_observe_accepts_sequence():
    lg = LoopGain()
    lg.observe(["e1", "e2", "e3"])  # magnitude = 3
    lg.observe(["e1", "e2"])  # magnitude = 2 → Aβ ≈ 0.67 → CONVERGING
    assert lg.state == CONVERGING


def test_observe_rejects_negative_number():
    lg = LoopGain()
    with pytest.raises(ValueError):
        lg.observe(-1.0)


def test_observe_rejects_nan():
    lg = LoopGain()
    with pytest.raises(ValueError):
        lg.observe(float("nan"))


def test_observe_rejects_unknown_type():
    lg = LoopGain()
    with pytest.raises(TypeError):
        lg.observe(object())


# ----- max_iterations safety cap -----


def test_max_iterations_triggers_terminal_state():
    lg = LoopGain(max_iterations=3)
    for _ in range(5):
        if not lg.should_continue():
            break
        lg.observe(10.0)
    # Constant errors will trigger OSCILLATING before MAX_ITERATIONS if window is large enough.
    # With max=3 and constant errors, we get OSCILLATING on iter 3 (Aβ=1.0 → OSCILLATING band).
    assert lg.state in (OSCILLATING, MAX_ITERATIONS)
    assert not lg.should_continue()


def test_max_iterations_with_converging_loop():
    """If max_iterations hits before convergence (and Aβ stays in a
    non-terminal band), the terminal state is MAX_ITERATIONS."""
    # Aβ = 0.5: clean CONVERGING band; cap at 2 forces MAX_ITERATIONS.
    lg = LoopGain(target_error=0.001, max_iterations=2)
    lg.observe(100.0)
    lg.observe(50.0)
    assert lg.state == MAX_ITERATIONS
    assert lg.result.outcome == "max_iterations"
    assert not lg.should_continue()


# ----- gain_margin -----


def test_gain_margin_greater_than_one_for_converging():
    lg = LoopGain(max_iterations=10)
    for e in _decay(0.5)[:5]:
        lg.observe(e)
    gm = lg.gain_margin
    assert gm is not None
    assert gm > 1.0


def test_gain_margin_none_before_observations():
    lg = LoopGain()
    assert lg.gain_margin is None


# ----- result before any observations -----


def test_result_not_started():
    lg = LoopGain()
    r = lg.result
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
    lg = LoopGain(target_error=0.5, assumed_fixed_cap=10)
    for e in [10.0, 1.0, 0.1]:
        if not lg.should_continue():
            break
        lg.observe(e)
    assert lg.result.savings_vs_fixed_cap > 0


def test_savings_zero_when_max_iterations_hit():
    lg = LoopGain(max_iterations=5, assumed_fixed_cap=10)
    for _ in range(10):
        if not lg.should_continue():
            break
        lg.observe(10.0)  # never converges
    result = lg.result
    if result.outcome == "max_iterations":
        assert result.savings_vs_fixed_cap == 0
