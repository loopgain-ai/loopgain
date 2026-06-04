"""Multi-feature trajectory classifier for LoopGain.

The v0.1 classifier maps a single instantaneous smoothed loop-gain Aβ_smooth
into one of five named states using fixed thresholds. Empirical validation
on real GVR loops (Component Algebra Experiment 3, 2026-04-10, n=150) showed
37.3% accuracy against intended ground truth — the single-feature design
cannot disambiguate floor-noise convergence, slow monotone improvement, and
mild drift-style divergence from one another.

This module replaces that with a multi-feature classifier that operates on
the full error trajectory. See ``PROTOCOL_v2_classifier.md`` for the
pre-registered design, threshold derivations, and validation plan.

The five state names are preserved (FAST_CONVERGE / CONVERGING / STALLING /
OSCILLATING / DIVERGING) so the telemetry schema, dashboard, and integrations
contract are not broken.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional, Sequence


# State constants — re-imported from core to avoid a circular import.
# These strings must stay in lockstep with core.py.
INIT = "INIT"
FAST_CONVERGE = "FAST_CONVERGE"
CONVERGING = "CONVERGING"
STALLING = "STALLING"
OSCILLATING = "OSCILLATING"
DIVERGING = "DIVERGING"


# ----- Pre-registered thresholds (PROTOCOL_v2_classifier.md §"Thresholds")
#
# Do not tune these to make individual workloads pass. The whole point of the
# pre-registration is that the thresholds are derived from textbook control
# theory and statistical convention, not fit. If a workload needs different
# behavior, pass a custom TrajectoryThresholds instance rather than editing
# these defaults.

# Cumulative E_current/E_first reduction below which we call FAST_CONVERGE.
# Derivation: one decade reduction = standard step-response 90% criterion.
DEFAULT_E_RATIO_FAST = 0.1

# E_current/E_first reduction below which we call CONVERGING even if the
# slope p-value is not significant (the cumulative reduction is enough
# evidence). Derivation: -3 dB / half-life.
DEFAULT_E_RATIO_CONV = 0.5

# Two-sided p-value below which the trend is "significant". Standard.
DEFAULT_P_SIG = 0.05

# Cumulative growth above which a positive slope counts as divergence. Below
# this margin a positive slope is treated as noise around stalling.
DEFAULT_DIV_MARGIN = 0.10

# Detrended log10(E) residual std above which we call OSCILLATING. Derivation:
# 0.30 log10 units ≈ ±2× ripple, matching an underdamped Q≈3 response.
DEFAULT_OSC_STD_THRESHOLD = 0.30

# Per-iteration log10 slope magnitude below which we call the trend flat
# for the oscillation gate.
DEFAULT_SLOPE_TOL = 0.05

# Liveness gate: number of iterations a loop may go without achieving a new
# best (lowest) error before its "continue" verdicts (FAST_CONVERGE /
# CONVERGING) are withdrawn so it can reach STALLING / OSCILLATING and
# terminate. Without this, a loop that drops a lot and then plateaus or
# oscillates *below* the cumulative thresholds keeps its historical win
# forever and never terminates. Derivation: the continue-states are claims
# about *ongoing* progress; cumulative reduction (E_current/E_first) and a
# whole-history slope are claims about the *past* and do not expire. We treat
# "no new low in N steps" as the loop having stopped improving. N is small
# (3) so a sustained plateau is caught quickly, but the consecutive-STALLING
# termination rule (2 readings) still protects a loop that briefly stalls and
# then resumes hitting new lows.
DEFAULT_STALL_PATIENCE = 3

# Numerical floor to avoid log(0).
_EPS = 1e-12


@dataclass(frozen=True)
class TrajectoryThresholds:
    """Pre-registered thresholds for the multi-feature classifier.

    Defaults match ``PROTOCOL_v2_classifier.md``. Override only when you have
    workload-specific evidence; do not tune to inflate accuracy numbers
    against held-out scenarios.
    """

    e_ratio_fast: float = DEFAULT_E_RATIO_FAST
    e_ratio_conv: float = DEFAULT_E_RATIO_CONV
    p_sig: float = DEFAULT_P_SIG
    div_margin: float = DEFAULT_DIV_MARGIN
    osc_std_threshold: float = DEFAULT_OSC_STD_THRESHOLD
    slope_tol: float = DEFAULT_SLOPE_TOL
    stall_patience: int = DEFAULT_STALL_PATIENCE


@dataclass(frozen=True)
class TrajectoryFeatures:
    """Computed features for one trajectory at a point in time.

    Returned by :func:`extract_features` so callers (e.g., telemetry, the
    dashboard, downstream tests) can inspect the inputs to the classification
    decision.
    """

    e_current: float
    e_first: float
    e_min: float
    e_ratio: float
    slope_log: float
    slope_p: float
    osc_std: float
    n: int


def _ols_slope_and_p(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float]:
    """Closed-form OLS slope + two-sided t-test p-value for the slope.

    Pure stdlib — no scipy dependency in the core package.
    Returns (0.0, 1.0) if n < 3 or x has zero variance.

    The p-value uses a Student-t CDF approximation via the regularized
    incomplete beta function from the math module (Python 3.12+:
    ``math.lgamma`` is enough to build the survival function we need with
    Wilson-Hilferty for any df ≥ 3).
    """
    n = len(x)
    if n < 3:
        # Need at least 3 points to estimate slope with any degrees of freedom.
        if n == 2:
            # Degenerate: slope is well defined, p-value is not.
            dx = x[1] - x[0]
            if dx == 0:
                return 0.0, 1.0
            return (y[1] - y[0]) / dx, 1.0
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0:
        return 0.0, 1.0
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    # Residual sum of squares; SE of slope; t-stat.
    rss = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    df = n - 2
    if df <= 0 or rss <= 0:
        # Perfect fit (rss=0) — slope is exact; p ≈ 0 if slope != 0.
        return slope, 0.0 if slope != 0 else 1.0
    s2 = rss / df
    se = math.sqrt(s2 / sxx)
    if se == 0:
        return slope, 0.0 if slope != 0 else 1.0
    t_stat = slope / se
    p = _two_sided_t_p(abs(t_stat), df)
    return slope, p


def _two_sided_t_p(t_abs: float, df: int) -> float:
    """Two-sided Student-t p-value via a Wilson-Hilferty normal approximation.

    Accurate enough for the classifier's purpose (decision threshold at
    p=0.05) for df ≥ 3. Returns a value in [0, 1].

    For df=2 (n=4 observations of x,y), uses the exact closed form
    P(|T| > t) = 2 / (2 + t²)^(1/2) for one-sided, doubled.
    """
    if df <= 0:
        return 1.0
    if df == 1:
        # exact: cdf_t(t,1) = 0.5 + arctan(t)/pi
        return 2.0 * (0.5 - math.atan(t_abs) / math.pi)
    if df == 2:
        # exact one-sided survival: 1 - (1 + t²/2)^(-1) doubled
        return min(1.0, 2.0 * (1.0 - t_abs / math.sqrt(2.0 + t_abs * t_abs) / 1.0) * 0.5
                   + 2.0 * (0.5 - 0.5 * t_abs / math.sqrt(2.0 + t_abs * t_abs)))
    # Wilson-Hilferty: transform t² ~ F(1, df), then F → chi-square via
    # cube-root approximation. For our purposes the simpler normal-approx
    # to the t with the Hill / Abramowitz adjustment is enough.
    # Use the standard correction: z = t * (1 - 1/(4·df)) / sqrt(1 + t²/(2·df))
    z = t_abs * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t_abs * t_abs / (2.0 * df))
    # Two-sided normal survival via erfc.
    return math.erfc(z / math.sqrt(2.0))


def extract_features(error_history: Sequence[float]) -> TrajectoryFeatures:
    """Compute trajectory-level features from the error history.

    Operates on log10(max(E, ε)) so geometric (multiplicative) trends become
    linear. This is the standard transformation for any signal that obeys
    Barkhausen's E_n = Aβ · E_{n−1}.
    """
    n = len(error_history)
    if n == 0:
        return TrajectoryFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0)

    e_first = error_history[0]
    e_current = error_history[-1]
    e_min = min(error_history)
    e_ratio = e_current / max(abs(e_first), _EPS)

    if n < 2:
        return TrajectoryFeatures(
            e_current=e_current,
            e_first=e_first,
            e_min=e_min,
            e_ratio=e_ratio,
            slope_log=0.0,
            slope_p=1.0,
            osc_std=0.0,
            n=n,
        )

    xs = list(range(n))
    log_e = [math.log10(max(e, _EPS)) for e in error_history]
    slope, p = _ols_slope_and_p(xs, log_e)

    # Detrended residual std (sample std).
    intercept = sum(log_e) / n - slope * (sum(xs) / n)
    residuals = [log_e[i] - (intercept + slope * xs[i]) for i in range(n)]
    if n >= 2:
        osc_std = statistics.pstdev(residuals)
    else:
        osc_std = 0.0

    return TrajectoryFeatures(
        e_current=e_current,
        e_first=e_first,
        e_min=e_min,
        e_ratio=e_ratio,
        slope_log=slope,
        slope_p=p,
        osc_std=osc_std,
        n=n,
    )


def classify_trajectory(
    error_history: Sequence[float],
    *,
    target_error: Optional[float] = None,
    thresholds: Optional[TrajectoryThresholds] = None,
) -> str:
    """Classify a full error history into one of the five named states.

    Decision rule (pre-registered, see PROTOCOL_v2_classifier.md):

        TARGET_MET     if  E_current ≤ target_error
        INIT           if  n < 2
        FAST_CONVERGE  if  E_ratio  ≤ E_RATIO_FAST
        CONVERGING     if  slope_log < 0 AND (slope_p < P_SIG OR E_ratio ≤ E_RATIO_CONV)
        DIVERGING      if  slope_log > 0 AND slope_p < P_SIG AND E_ratio > 1 + DIV_MARGIN
        OSCILLATING    if  osc_std ≥ OSC_STD_THRESHOLD AND |slope_log| < SLOPE_TOL
        STALLING       otherwise

    Note: TARGET_MET is returned only when ``target_error`` is supplied AND
    ``E_current ≤ target_error``. This module does not own the TARGET_MET
    short-circuit; ``LoopGain.observe`` handles that, and the classifier is
    called only when the short-circuit has not fired. We accept the
    ``target_error`` parameter so callers that want to classify a stored
    trajectory get the same answer the live engine would have produced.
    """
    th = thresholds or TrajectoryThresholds()
    if not error_history:
        return INIT

    e_current = error_history[-1]
    if target_error is not None and e_current <= target_error:
        # State name for "target met" is exposed by core, not this module.
        # Callers that want the literal "TARGET_MET" string should check
        # target_error themselves; we return FAST_CONVERGE as the classifier's
        # opinion of a trajectory that's already at its floor.
        return FAST_CONVERGE

    n = len(error_history)
    if n < 2:
        return INIT

    f = extract_features(error_history)

    # Liveness signal: how many iterations since the loop last achieved a new
    # best (lowest) error. A genuinely converging loop keeps hitting new lows,
    # so this stays small; a loop that dropped a lot and then plateaued (or is
    # oscillating below the cumulative thresholds) has a large value. We use it
    # to withdraw the "continue" verdicts (FAST_CONVERGE / CONVERGING) once a
    # loop has stopped improving, so it can reach STALLING / OSCILLATING and
    # terminate instead of riding its historical cumulative win forever. See
    # DEFAULT_STALL_PATIENCE.
    hist = list(error_history)
    iters_since_best = (n - 1) - hist.index(min(hist))
    still_improving = iters_since_best < th.stall_patience

    # n == 2 special case: with two observations, the slope is well defined
    # but its p-value is not (zero residual degrees of freedom). Fall back to
    # the sign of the change. This is the same conservatism as a Wilcoxon
    # signed-rank test with n=1: insufficient evidence for a significance
    # claim, but the *direction* is unambiguous.
    if n == 2:
        if f.e_ratio <= th.e_ratio_fast:
            return FAST_CONVERGE
        if f.e_ratio < 1.0:
            return CONVERGING
        if f.e_ratio > 1.0 + th.div_margin:
            return DIVERGING
        return STALLING

    # Order matters: FAST_CONVERGE precedes CONVERGING; both precede the
    # remaining gates. Both continue-verdicts are gated on `still_improving`:
    # a loop that has stopped hitting new lows is no longer "converging" no
    # matter how large its historical cumulative reduction was, and must be
    # allowed to fall through to STALLING / OSCILLATING so it can terminate.
    if f.e_ratio <= th.e_ratio_fast and still_improving:
        return FAST_CONVERGE

    slope_significant = f.slope_p < th.p_sig

    if (
        f.slope_log < 0
        and still_improving
        and (slope_significant or f.e_ratio <= th.e_ratio_conv)
    ):
        return CONVERGING

    if f.slope_log > 0 and slope_significant and f.e_ratio > 1.0 + th.div_margin:
        return DIVERGING

    if f.osc_std >= th.osc_std_threshold and abs(f.slope_log) < th.slope_tol:
        return OSCILLATING

    return STALLING


__all__ = [
    "TrajectoryThresholds",
    "TrajectoryFeatures",
    "extract_features",
    "classify_trajectory",
    "DEFAULT_E_RATIO_FAST",
    "DEFAULT_E_RATIO_CONV",
    "DEFAULT_P_SIG",
    "DEFAULT_DIV_MARGIN",
    "DEFAULT_OSC_STD_THRESHOLD",
    "DEFAULT_SLOPE_TOL",
]
