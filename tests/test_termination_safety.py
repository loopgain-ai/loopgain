"""Termination-safety tests: a loop must not run unbounded.

Regression coverage for the FAST_CONVERGE/CONVERGING liveness bug (2026-06):
the trajectory classifier used *cumulative* reduction (E_current/E_first) and a
*whole-history* slope to emit the "continue" verdicts FAST_CONVERGE and
CONVERGING. A loop that reduced its error and then plateaued (or oscillated)
*below* the cumulative thresholds kept its historical win forever — it was
pinned in a continue-state, never reached STALLING/OSCILLATING, and with the
(then-default) max_iterations=None it ran forever.

The fix has two independent layers, each tested here:
  1. A liveness gate on the continue-verdicts: a loop that has not achieved a
     new best error in `stall_patience` iterations is no longer treated as
     "improving", so it can reach STALLING/OSCILLATING and terminate.
  2. A bounded default max_iterations backstop, so the library can never run
     truly unbounded even if a future classifier path regresses.

Output quality was never at risk (best-so-far rollback held the good answer);
the bug was a *liveness* failure — the loop never returned to hand it back.
"""

from __future__ import annotations

import pytest

from loopgain import CONVERGING, FAST_CONVERGE, LoopGain, classify_trajectory

# Hard test guard: large enough that a *correctly* terminating loop never hits
# it, small enough that a regression (unbounded loop) fails fast instead of
# hanging the suite.
GUARD = 500


def _run_to_termination(lg: LoopGain, errors, guard: int = GUARD):
    """Drive a loop, plateauing/repeating the last error, until it terminates
    or hits the guard. Returns (iterations_run, hit_guard)."""
    i = 0
    while lg.should_continue():
        e = errors[i] if i < len(errors) else errors[-1]
        lg.observe(e, output=f"o{i}")
        i += 1
        if i >= guard:
            return i, True
    return i, False


# ----- Layer 1: classifier liveness gate -----


def test_plateau_below_fast_floor_terminates_without_max_iter():
    """Error drops to 8% of initial then plateaus. e_ratio<=0.1 used to pin
    FAST_CONVERGE forever. Must now terminate via STALLING."""
    lg = LoopGain(max_iterations=None, target_error=None)
    n, hit_guard = _run_to_termination(lg, [100, 8, 8, 8, 8, 8, 8, 8])
    assert not hit_guard, f"loop did not terminate within {GUARD} iters (unbounded)"
    assert not lg.should_continue()
    assert lg.result.best_error == 8.0  # best-so-far still returned


def test_plateau_above_fast_floor_terminates_without_max_iter():
    """Error drops to 30% of initial (below E_RATIO_CONV=0.5) then plateaus.
    e_ratio<=0.5 with a whole-history negative slope used to pin CONVERGING
    forever. Must now terminate."""
    lg = LoopGain(max_iterations=None, target_error=None)
    n, hit_guard = _run_to_termination(lg, [100, 30, 30, 30, 30, 30, 30, 30])
    assert not hit_guard, f"loop did not terminate within {GUARD} iters (unbounded)"
    assert not lg.should_continue()


def test_oscillation_below_floor_terminates_without_max_iter():
    """Oscillation entirely below the 10% cumulative floor used to be shadowed
    by FAST_CONVERGE. Must now terminate (OSCILLATING or STALLING)."""
    lg = LoopGain(max_iterations=None, target_error=None)
    n, hit_guard = _run_to_termination(lg, [100, 5, 8, 5, 8, 5, 8, 5, 8])
    assert not hit_guard, f"loop did not terminate within {GUARD} iters (unbounded)"
    assert not lg.should_continue()


def test_classifier_flags_plateau_after_big_drop_as_terminable():
    """Direct classifier check: a big drop followed by a flat tail must NOT be
    reported as a continue-state (FAST_CONVERGE/CONVERGING)."""
    plateau_low = [100, 8, 8, 8, 8, 8]
    plateau_mid = [100, 30, 30, 30, 30, 30]
    assert classify_trajectory(plateau_low) not in (FAST_CONVERGE, CONVERGING)
    assert classify_trajectory(plateau_mid) not in (FAST_CONVERGE, CONVERGING)


def test_genuine_fast_converge_still_continues():
    """Guard against over-correction: a monotone steep decline that keeps
    hitting new lows must still read FAST_CONVERGE (continue), not be
    prematurely stalled."""
    monotone = [100, 25, 6, 1.5, 0.4, 0.1]  # new low every step
    assert classify_trajectory(monotone) == FAST_CONVERGE


def test_genuine_converging_still_continues():
    """A steady decline landing between the two cumulative thresholds must
    still read CONVERGING while it is still hitting new lows."""
    converging = [10.0, 8.0, 6.4, 5.1, 4.1, 3.3]  # ~0.8x/step, new low every step
    assert classify_trajectory(converging) == CONVERGING


# ----- Layer 2: bounded default backstop -----


def test_default_max_iterations_is_a_bounded_backstop():
    """The default config must not be able to run unbounded. A never-improving
    loop under all-default construction must terminate at the backstop."""
    lg = LoopGain()  # all defaults
    assert lg.max_iterations is not None, "default max_iterations must be bounded"
    # A strictly increasing error never converges/stalls into best-so-far early
    # under every classifier path; the backstop must still stop it.
    i, hit_guard = _run_to_termination(lg, list(range(1, GUARD + 5)))
    assert not hit_guard, "default backstop failed to bound the loop"
    assert not lg.should_continue()
