"""Optimized gates must retain exact policy and public feature semantics."""
import math
import statistics
from dataclasses import replace

import pytest

import loopgain.classifier as classifier


@pytest.mark.parametrize('history,expected', [
    ([10.0, 0.5], classifier.FAST_CONVERGE),
    ([10.0, 1.0], classifier.FAST_CONVERGE),
    ([10.0, math.nextafter(1.0, math.inf)], classifier.CONVERGING),
    ([10.0, 9.0], classifier.CONVERGING),
    ([10.0, 10.0], classifier.STALLING),
    ([10.0, 11.0], classifier.STALLING),
    ([10.0, 10.0 * math.nextafter(1.1, math.inf)], classifier.DIVERGING),
    ([0.0, 0.0], classifier.FAST_CONVERGE),
    ([0.0, 1.0], classifier.DIVERGING),
])
def test_two_points_do_not_fit_a_regression(monkeypatch, history, expected):
    def unexpected(*args, **kwargs):
        raise AssertionError('two-point ratio gate does not require regression')
    monkeypatch.setattr(classifier, '_ols_slope_and_p', unexpected)
    assert classifier.classify_trajectory(history) == expected
    assert classifier.classify_trajectory(tuple(history)) == expected


def test_fast_cumulative_gate_needs_no_regression(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('decisive cumulative gate does not require regression')
    monkeypatch.setattr(classifier, '_ols_slope_and_p', unexpected)
    assert classifier.classify_trajectory([100, 50, 20, 5]) == classifier.FAST_CONVERGE


def test_historical_reduction_does_not_bypass_liveness():
    assert classifier.classify_trajectory([100, 5, 5, 5, 5]) == classifier.STALLING


@pytest.mark.parametrize('history,expected', [
    ([100 * .95 ** i for i in range(20)], classifier.CONVERGING),
    ([100 * 1.05 ** i for i in range(20)], classifier.DIVERGING),
])
def test_decisive_trend_does_not_compute_variance(monkeypatch, history, expected):
    def unexpected(*args, **kwargs):
        raise AssertionError('decisive trend does not require residual variance')
    monkeypatch.setattr(classifier.statistics, 'pstdev', unexpected)
    assert classifier.classify_trajectory(history) == expected


def test_flat_trend_still_computes_exact_variance(monkeypatch):
    calls = []
    original = statistics.pstdev
    def recording(values):
        calls.append(tuple(values))
        return original(values)
    monkeypatch.setattr(classifier.statistics, 'pstdev', recording)
    assert classifier.classify_trajectory([2.0] * 8) == classifier.STALLING
    assert len(calls) == 1


@pytest.mark.parametrize('history', [
    [2.0] * 8,
    [1, 10, 2, 20, 3, 15],
    [1e-300] * 8,
    [1e300, 1e-300] * 4,
    [10 ** (200 + .002 * i) for i in range(10)],
])
def test_public_features_preserve_exact_population_std(history):
    features = classifier.extract_features(history)
    log_e = [math.log10(max(e, 1e-12)) for e in history]
    xs = list(range(len(history)))
    slope, p = classifier._ols_slope_and_p(xs, log_e)
    intercept = sum(log_e) / len(log_e) - slope * (sum(xs) / len(xs))
    residuals = [log_e[i] - (intercept + slope * xs[i]) for i in xs]
    assert features.slope_log == slope
    assert features.slope_p == p
    assert features.osc_std == statistics.pstdev(residuals)


def test_exact_custom_oscillation_boundary_is_preserved():
    history = [1, 10, 2, 20, 3, 15]
    features = classifier.extract_features(history)
    thresholds = classifier.TrajectoryThresholds(
        e_ratio_fast=-1, e_ratio_conv=-1, p_sig=-1,
        slope_tol=1000, osc_std_threshold=features.osc_std,
    )
    assert classifier.classify_trajectory(history, thresholds=thresholds) == classifier.OSCILLATING
    above = replace(thresholds, osc_std_threshold=math.nextafter(features.osc_std, math.inf))
    assert classifier.classify_trajectory(history, thresholds=above) == classifier.STALLING


@pytest.mark.parametrize('history', [
    [float('inf'), 1.0], [1.0, float('inf')],
    [float('nan'), 1.0], [1.0, float('nan')],
    [100, float('inf'), 1], [100, float('nan'), 1],
])
def test_nonfinite_direct_inputs_keep_eager_feature_error(history):
    # statistics.pstdev's exception type differs across supported Python
    # versions. Match the unchanged eager feature API on this interpreter.
    with pytest.raises(Exception) as reference:
        classifier.extract_features(history)
    with pytest.raises(type(reference.value)) as actual:
        classifier.classify_trajectory(history)
    assert str(actual.value) == str(reference.value)


def test_large_finite_integer_input_does_not_coerce_to_float_for_finiteness():
    value = 10 ** 1000
    assert classifier.classify_trajectory([value, value // 2]) == classifier.CONVERGING
