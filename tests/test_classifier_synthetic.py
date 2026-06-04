"""Tier-1 synthetic validation of the trajectory classifier.

Implements the synthetic trajectories enumerated in
``PROTOCOL_v2_classifier.md`` §"Tier 1 — Synthetic trajectories." Each
regime has a known mathematical ground truth: the classifier's job is to
recover the ground-truth label from a noise-controlled sample of the
trajectory.

Pass criterion (pre-registered):
    - Deterministic regimes: 100% correct.
    - Noisy regimes: ≥ 95% correct over 100 random seeds per regime.

Failures here mean the classifier's math is wrong, not that the workload is
unusual. Synthetic tests are the gate that must pass before any real-LLM
validation is taken seriously.
"""

from __future__ import annotations

import math
import random

import pytest

from loopgain import (
    CONVERGING,
    DIVERGING,
    FAST_CONVERGE,
    OSCILLATING,
    STALLING,
    TrajectoryFeatures,
    TrajectoryThresholds,
    classify_trajectory,
    extract_features,
)
from loopgain.classifier import _ols_slope_and_p


# ----- OLS slope / p-value building blocks -----


def test_ols_slope_zero_residual_zero_p():
    """Perfect linear fit: slope is exact, p-value drops to 0 (highly
    significant) when slope != 0."""
    xs = [0, 1, 2, 3, 4]
    ys = [0.0, 1.0, 2.0, 3.0, 4.0]  # slope = 1, intercept = 0, rss = 0
    slope, p = _ols_slope_and_p(xs, ys)
    assert slope == pytest.approx(1.0)
    assert p == 0.0


def test_ols_slope_constant_y_no_significance():
    """Constant y: slope is exactly 0, p-value 1 (no evidence of trend)."""
    xs = [0, 1, 2, 3, 4]
    ys = [3.7, 3.7, 3.7, 3.7, 3.7]
    slope, p = _ols_slope_and_p(xs, ys)
    assert slope == pytest.approx(0.0)
    assert p == 1.0


def test_ols_slope_pure_noise_no_significance():
    """Random noise with no real trend: p > 0.05 for typical samples."""
    random.seed(42)
    n_iter = 100
    n_significant = 0
    for _ in range(n_iter):
        xs = list(range(8))
        ys = [random.gauss(0.0, 1.0) for _ in xs]
        _, p = _ols_slope_and_p(xs, ys)
        if p < 0.05:
            n_significant += 1
    # Type-I error should sit near the 5% nominal rate. Allow generous slack
    # for the t-approximation: anywhere in [0%, 20%] over 100 trials is fine.
    assert n_significant <= 20


def test_ols_slope_strong_signal_is_significant():
    """Clear linear trend with mild noise: p < 0.05."""
    random.seed(0)
    xs = list(range(10))
    ys = [0.5 * x + random.gauss(0, 0.1) for x in xs]
    _, p = _ols_slope_and_p(xs, ys)
    assert p < 0.05


# ----- Deterministic synthetic regimes (100% accuracy expected) -----


@pytest.mark.parametrize("r", [0.5, 0.7, 0.85])
def test_clean_geometric_convergence(r):
    """E_n = E_0 · r^n with r ∈ {0.5, 0.7, 0.85}.

    r=0.5 reduces by 8× over 4 steps → e_ratio=0.0625 → FAST_CONVERGE.
    r=0.7 reduces by 2.4× over 4 steps → e_ratio=0.4 → CONVERGING.
    r=0.85 reduces by 1.55× → e_ratio=0.64 → still CONVERGING (slope
    significant after 8 steps).
    """
    n = 8
    trajectory = [10.0 * (r ** i) for i in range(n)]
    state = classify_trajectory(trajectory)
    assert state in (CONVERGING, FAST_CONVERGE), (
        f"r={r}: expected converging family, got {state}, "
        f"features={extract_features(trajectory)}"
    )


@pytest.mark.parametrize("r", [0.05, 0.1, 0.2])
def test_fast_convergence(r):
    """E_n = E_0 · r^n with r ≤ 0.2 reduces by at least one decade in
    2-3 steps → FAST_CONVERGE."""
    trajectory = [10.0 * (r ** i) for i in range(5)]
    state = classify_trajectory(trajectory)
    assert state == FAST_CONVERGE, (
        f"r={r}: expected FAST_CONVERGE, got {state}, "
        f"features={extract_features(trajectory)}"
    )


@pytest.mark.parametrize("r", [1.1, 1.3, 1.5])
def test_clean_divergence(r):
    """E_n = E_0 · r^n with r > 1 grows monotonically → DIVERGING."""
    trajectory = [10.0 * (r ** i) for i in range(6)]
    state = classify_trajectory(trajectory)
    assert state == DIVERGING, (
        f"r={r}: expected DIVERGING, got {state}, "
        f"features={extract_features(trajectory)}"
    )


def test_pure_oscillation_around_fixed_point():
    """E_n alternates around a fixed point with no net trend → OSCILLATING.

    Construction: odd-length trajectory so first and last values coincide;
    that zeroes the OLS slope exactly. Amplitude is ±log10(5)/2 ≈ 0.35 on
    the detrended log10 residuals, comfortably above the 0.30 threshold.
    """
    base = 10.0
    # n=11, odd-symmetric: [50, 10, 50, 10, 50, 10, 50, 10, 50, 10, 50]
    trajectory = [base * (5.0 if i % 2 == 0 else 1.0) for i in range(11)]
    f = extract_features(trajectory)
    state = classify_trajectory(trajectory)
    assert state == OSCILLATING, (
        f"expected OSCILLATING, got {state}, "
        f"slope={f.slope_log:.3f}, osc_std={f.osc_std:.3f}, "
        f"e_ratio={f.e_ratio:.3f}"
    )


def test_pure_stall_no_trend():
    """E_n = constant + small noise → STALLING (no trend, no oscillation)."""
    random.seed(7)
    base = 5.0
    trajectory = [base + random.gauss(0, 0.05) for _ in range(8)]
    state = classify_trajectory(trajectory)
    assert state == STALLING, (
        f"expected STALLING, got {state}, "
        f"features={extract_features(trajectory)}"
    )


def test_floor_convergence_already_flat_at_floor_stalls():
    """A loop already pinned at the numerical floor from iteration 0, flat,
    classifies as STALLING — not FAST_CONVERGE.

    Updated 2026-06 with the liveness-gate fix (see DEFAULT_STALL_PATIENCE).
    Previously this returned FAST_CONVERGE on the strength of cumulative
    reduction alone — but FAST_CONVERGE is a *continue* verdict, so an
    at-floor flat loop would have continued (and, with no max_iterations,
    run unbounded) instead of stopping. STALLING is the correct verdict: the
    loop has made no progress for `stall_patience` iterations, so it
    terminates via the consecutive-stall rule and returns best-so-far (the
    floor value — a fine answer). In real use the `target_error`
    short-circuit (next test) handles the at-target case directly."""
    trajectory = [1e-15] * 5
    state = classify_trajectory(trajectory)
    assert state == STALLING


def test_target_met_short_circuit():
    """Passing target_error returns FAST_CONVERGE when current ≤ target."""
    trajectory = [10.0, 5.0, 0.4]
    state = classify_trajectory(trajectory, target_error=0.5)
    assert state == FAST_CONVERGE


# ----- Noisy synthetic regimes (≥ 95% accuracy expected) -----


def _run_seeds(generator, expected, n_seeds=100):
    """Run ``generator(seed)`` for n_seeds and report how many classify as
    expected. Returns (n_correct, total, mismatches[:10])."""
    n_correct = 0
    mismatches = []
    for seed in range(n_seeds):
        traj = generator(seed)
        state = classify_trajectory(traj)
        if state == expected:
            n_correct += 1
        elif len(mismatches) < 10:
            mismatches.append(
                (seed, state, extract_features(traj))
            )
    return n_correct, n_seeds, mismatches


def test_noisy_clean_convergence_95pct():
    """E_n = E_0 · 0.7^n + 10% multiplicative noise.

    Pre-registered pass: ≥ 95% classify as CONVERGING (or FAST_CONVERGE if
    the cumulative reduction crosses one decade — semantically equivalent).
    """
    def gen(seed):
        rng = random.Random(seed)
        n = 8
        traj = []
        for i in range(n):
            clean = 10.0 * (0.7 ** i)
            noisy = clean * (1.0 + rng.gauss(0, 0.1))
            traj.append(max(noisy, 1e-6))
        return traj

    n_correct = 0
    mismatches = []
    for seed in range(100):
        traj = gen(seed)
        state = classify_trajectory(traj)
        if state in (CONVERGING, FAST_CONVERGE):
            n_correct += 1
        elif len(mismatches) < 5:
            mismatches.append((seed, state))
    assert n_correct >= 95, (
        f"only {n_correct}/100 noisy-converging trials classified correctly; "
        f"first mismatches: {mismatches}"
    )


def test_noisy_divergence_95pct():
    """E_n = E_0 · 1.2^n + 10% multiplicative noise. ≥ 95% DIVERGING."""
    def gen(seed):
        rng = random.Random(seed)
        n = 8
        traj = []
        for i in range(n):
            clean = 10.0 * (1.2 ** i)
            noisy = clean * (1.0 + rng.gauss(0, 0.1))
            traj.append(max(noisy, 1e-6))
        return traj

    n_correct, total, mismatches = _run_seeds(gen, DIVERGING, n_seeds=100)
    assert n_correct >= 95, (
        f"only {n_correct}/{total} noisy-diverging trials classified DIVERGING; "
        f"first mismatches: {[(s, st) for s, st, _ in mismatches]}"
    )


def test_noisy_stall_95pct():
    """Pure stall + 10% multiplicative noise (no underlying trend).
    ≥ 95% STALLING.
    """
    def gen(seed):
        rng = random.Random(seed)
        base = 5.0
        return [max(base * (1.0 + rng.gauss(0, 0.10)), 1e-6) for _ in range(8)]

    n_correct, total, mismatches = _run_seeds(gen, STALLING, n_seeds=100)
    assert n_correct >= 95, (
        f"only {n_correct}/{total} noisy-stall trials classified STALLING; "
        f"first mismatches: {[(s, st) for s, st, _ in mismatches]}"
    )


# ----- Boundary tests -----


def test_slow_convergence_with_cumulative_evidence():
    """Cumulative E_ratio ≤ 0.5 should trigger CONVERGING even if the per-step
    slope p-value is borderline. This is the "trust the cumulative
    reduction" branch."""
    trajectory = [10.0, 9.0, 8.0, 7.0, 5.0]  # E_ratio = 0.5
    state = classify_trajectory(trajectory)
    assert state == CONVERGING


def test_small_growth_does_not_trigger_divergence():
    """E_ratio = 1.05 (5% growth) is below DIV_MARGIN=0.10 → not DIVERGING.
    Should be STALLING (no significant trend large enough to call it)."""
    trajectory = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
    state = classify_trajectory(trajectory)
    assert state == STALLING


def test_large_growth_with_significance_triggers_divergence():
    """E_ratio > 1.1 with significant positive slope → DIVERGING."""
    trajectory = [10.0, 11.0, 12.5, 14.0, 16.0, 18.5]
    state = classify_trajectory(trajectory)
    assert state == DIVERGING


def test_n_equals_2_clear_improvement_is_converging():
    """With n=2, a clear E_ratio < 1 returns CONVERGING via the sign-of-
    difference fallback (slope p-value undefined at n=2)."""
    trajectory = [10.0, 5.0]
    state = classify_trajectory(trajectory)
    assert state == CONVERGING


def test_n_equals_2_clear_degradation_is_diverging():
    """With n=2, a clear E_ratio > 1 + DIV_MARGIN returns DIVERGING."""
    trajectory = [10.0, 12.0]
    state = classify_trajectory(trajectory)
    assert state == DIVERGING


def test_n_equals_2_tiny_change_is_stalling():
    """With n=2, a change inside (1-FAST, 1+DIV_MARGIN] is STALLING."""
    trajectory = [10.0, 10.05]
    state = classify_trajectory(trajectory)
    assert state == STALLING


# ----- Threshold customization is honored -----


def test_custom_thresholds_override_defaults():
    """Tighter FAST threshold pushes a marginal trajectory into CONVERGING."""
    trajectory = [10.0, 1.5, 0.5, 0.15]  # e_ratio = 0.015 vs default 0.1
    thr_tight = TrajectoryThresholds(e_ratio_fast=0.001)
    assert classify_trajectory(trajectory, thresholds=thr_tight) == CONVERGING
    # Default threshold sees 0.015 ≤ 0.1 → FAST_CONVERGE.
    assert classify_trajectory(trajectory) == FAST_CONVERGE
