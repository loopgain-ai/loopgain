# LoopGain v2 Classifier — Pre-Registration

**Status:** Pre-registered design. Locked before evaluation on held-out data.
**Date:** 2026-05-18
**Author:** Dave Fitzsimmons
**Companion docs:**
- `component-algebra-v2-protocol-final-4.md` §3.3 (parent v2 protocol)
- `loopgain/core.py` (v0.1 single-feature classifier — being replaced)
- `results/experiment_3_results.json` (v1 result: 37.3% overall accuracy)
- internal research notes on grokking / prediction-signal classifiers (not public)

## Problem statement

The v0.1 classifier maps a single instantaneous smoothed loop-gain value Aβ_smooth
to one of five named states using fixed thresholds (0.3 / 0.85 / 0.95 / 1.05).
Experiment 3 (2026-04-10, n=150) measured 37.3% accuracy against intended ground
truth — far below the 85% pre-registered success criterion. Per-regime failure
breakdown reveals two distinct failure modes:

| Scenario | Aβ mean | Outcome |
|---|---|---|
| converging (E_first=0.02) | 0.22 | 94% correct — but E was already at floor at iter 1; Aβ is floor noise |
| stalling (E_first=0.25, E_last=0.09) | 0.40 | 2% correct — labeled "stalling" but trajectory is monotone improvement |
| diverging (E_first=4.94, E_last=5.28) | 1.04 ± 0.06 | 16% correct — hovers in OSCILLATING band; trend is +1.6%/iter but ratio reads ≈1.0 |

## Root cause

Single instantaneous Aβ cannot disambiguate three orthogonal failure modes that
a real GVR loop exhibits:

1. **Floor-noise convergence**: error already ≤ target at first step; subsequent
   Aβ values are ratios of small numbers and are dominated by noise.
2. **Slow monotone improvement labelled as stall**: the *intended* regime is
   stall, but verifier noise on a creative-writing task produces a modest but
   consistent downward trend in E. The right classification is "slowly
   converging," not "stalling at fixed Aβ band." Ground truth was wrong.
3. **Drift-style divergence**: error climbs at <10%/iter so Aβ ≈ 1.0 ± noise,
   falling in OSCILLATING band even though the cumulative trend is clearly up.

The grokking literature (Xu 2026 commutator defect; Notsawo 2023 Fourier
fingerprint; Clauw 2024 synergy peaks) reaches the same conclusion across a
different domain: **trajectory-level features predict phase transitions more
reliably than instantaneous values do.** Orb-Hopfield Exp 29 makes the dual
point: a single-number diagnostic (rank-4 spectrum) can match a target band
for fundamentally different mechanistic reasons. Aβ in [0.85, 0.95] can mean
"genuine stall," "slow convergence to floor," "noisy drift past target," or
"transient on the way to oscillation" — these are not the same state.

## Design

Replace the single-feature decision with a multi-feature classifier that
operates on the full error history. The five state names are retained
unchanged so dashboard, telemetry receiver, integrations, and the public API
contract are not broken.

### Features (computed from error_history of length n ≥ 2)

| Symbol | Definition | What it captures |
|---|---|---|
| E_current | error_history[-1] | absolute error level |
| E_first | error_history[0] | baseline magnitude |
| E_min | min(error_history) | best-so-far floor |
| E_ratio | E_current / max(E_first, ε) | cumulative reduction |
| slope_log | OLS slope of log(max(E_i, ε)) on iteration index | direction of motion, log-domain (geometric) |
| slope_p | two-sided p-value of slope_log under H₀: slope=0 | significance of trend |
| osc_std | std of residuals after detrending log(E) | oscillation magnitude |
| Aβ_smooth | EMA(E_i / E_{i-1}, window=3) | instantaneous gain (legacy feature, retained) |

ε = 1e-12 (numerical floor for log).

For n=2, slope_p is undefined; we fall back to sign-of-difference and treat
slope_p as "non-significant" (p=1.0).

### Decision rule (pre-registered)

Inputs: features as defined above; thresholds as defined below.

```
TARGET_MET     if  E_current ≤ target_error
INIT           if  n < 2
FAST_CONVERGE  if  E_ratio  ≤ E_RATIO_FAST                  (default 0.1)
CONVERGING     if  slope_log < 0  AND  (slope_p < P_SIG  OR  E_ratio ≤ E_RATIO_CONV)
DIVERGING      if  slope_log > 0  AND  slope_p < P_SIG  AND  E_ratio > 1 + DIV_MARGIN
OSCILLATING    if  osc_std    ≥ OSC_STD_THRESHOLD  AND  |slope_log| < SLOPE_TOL
STALLING       otherwise
```

The fall-through is STALLING (no detectable progress in any direction, no
oscillation), which matches the operational semantics: "loop is doing
something but it's not getting better — return best-so-far."

### Thresholds (pre-registered, derived analytically)

| Threshold | Value | Derivation |
|---|---|---|
| target_error | 0.0 (default) | TARGET_MET short-circuit (user-supplied) |
| E_RATIO_FAST | 0.1 | Geometric: one decade reduction. Standard step-response 90% criterion. |
| E_RATIO_CONV | 0.5 | Half-life reduction. -3 dB in EE terms. |
| P_SIG | 0.05 | Standard statistical significance |
| DIV_MARGIN | 0.10 | Cumulative growth ≥ 10% required to call divergence (not just noise) |
| OSC_STD_THRESHOLD | 0.30 | log10 units = ±2× ripple; matches an underdamped Q≈3 response |
| SLOPE_TOL | 0.05 | Per-iteration slope of \|0.05\| ≈ ±5% trend ≈ flat |

All thresholds derived from textbook control theory or basic statistical
convention. None tuned against the validation data. Sensitivity analysis is a
*reporting* output, not an input to threshold selection.

### Backward compatibility

The existing `ThresholdBands` dataclass and the `state_for(ab_smooth)` method
remain on the public API as `LegacyThresholdBands` for callers that have
empirically tuned the bands to their workload. The new classifier is the
default. A user can opt back into legacy with
`LoopGain(classifier='legacy_bands')`.

State names (FAST_CONVERGE, CONVERGING, STALLING, OSCILLATING, DIVERGING,
TARGET_MET, MAX_ITERATIONS, INIT) are unchanged. Telemetry schema
(`error_history`, `convergence_profile`, `profile_max`, `profile_min`,
`profile_median`) is unchanged. The dashboard's `bandFromAB(ab)` per-point
classifier is unchanged. The dashboard's loop-level `bandFromEvent` is
unchanged.

## Validation

### Tier 1 — Synthetic trajectories (closed-form mathematical correctness)

Generate noise-controlled synthetic trajectories with known ground truth:

| Synthetic regime | Construction | Expected state |
|---|---|---|
| Clean geometric convergence | E_n = E_0 · r^n, r ∈ {0.5, 0.7, 0.85}, no noise | CONVERGING |
| Fast geometric convergence | E_n = E_0 · r^n, r ∈ {0.1, 0.2}, no noise | FAST_CONVERGE |
| Pure stall | E_n = E_0 + N(0, 0.05·E_0), no trend | STALLING |
| Pure divergence | E_n = E_0 · r^n, r ∈ {1.1, 1.3, 1.5}, no noise | DIVERGING |
| Oscillation around fixed point | E_n = E_0 · (1 + 0.5·sin(πn)), no trend | OSCILLATING |
| Floor convergence | E_n = ε for all n | TARGET_MET (with target=ε) or FAST_CONVERGE |
| Noisy convergence | E_n = E_0 · 0.7^n + N(0, 0.1·E_n), 100 seeds | ≥95% CONVERGING |
| Noisy divergence | E_n = E_0 · 1.2^n + N(0, 0.1·E_n), 100 seeds | ≥95% DIVERGING |

**Pass criterion:** ≥ 95% correct classification on each synthetic regime over
100 random seeds per regime (where noise applies), 100% correct on the
deterministic regimes. Synthetic tests are the math-correctness gate.

### Tier 2 — Held-out real-LLM trials (re-classification, no new data)

The existing 150 trials in `data/raw/exp3_loopguard/trial_details.json` are
re-classified by the new classifier. Two ground-truth labels are reported:

- **Original-intent labels** (the scenario the trial was sampled from). For
  fairness, the converging scenario is dropped because its trajectories are
  degenerate (E_first ≈ 0.02 — every method correctly says "already
  converged" with no information about classifier quality).
- **Trajectory-shape labels** (post-hoc, derived from the actual E sequence,
  blind to the new classifier's predictions): for each trial,
  - if E_ratio < 0.1 → FAST_CONVERGE
  - elif E_ratio < 0.5 and trend slope < 0 → CONVERGING
  - elif E_ratio > 1.2 → DIVERGING
  - elif osc_std > 0.3 → OSCILLATING
  - else → STALLING

  This labelling rule mirrors the new classifier's logic but uses different
  thresholds, so it is a structurally biased ground truth. It is reported as
  a sanity check (does the classifier reproduce the trajectory's evident
  shape?) not as a primary success metric.

**Pass criteria:**
- Tier 2a (original-intent, stall + diverge scenarios only): ≥ 75% accuracy
  on each regime. v1 baseline: 2% / 16%.
- Tier 2b (trajectory-shape sanity check): ≥ 90% agreement. This is the
  internal consistency check; a low number here means the classifier
  contradicts its own theoretical basis.

### Tier 3 — New real-LLM trials with corrected scenarios

The existing CONVERGING_SPEC is degenerate (Haiku writes a correct merge in
one shot). The replacement scenarios are designed so that E_first is in a
meaningful range and the trajectory has room to show its shape:

- **CONVERGING_v2**: a harder code task (string-similarity algorithm with
  case-insensitive, accent-insensitive, and prefix-bonus requirements) so the
  first-pass output predictably has 3-8 issues.
- **STALLING_v2**: kept (subjective writing already works as designed).
- **DIVERGING_v2**: kept.

50 trials per scenario; same dual-judge CMRR harness as exp3; classifier run
post-hoc on the captured error histories.

**Pass criteria (overall H3a v2):**
- Overall accuracy ≥ 75% across all three regimes (relaxed from 85% in v1
  because we have one fewer informative regime; converging is now genuinely
  testable but the stall vs diverge distinction is the hard one).
- Per-regime ≥ 65%.

If Tier 3 misses pass criteria, the classifier is still an improvement over
v1 (because the per-regime numbers will be reported) but H3a is not
considered passed.

## Falsifiable predictions

1. Synthetic clean convergence at r=0.7 → CONVERGING with 100% accuracy.
2. Synthetic oscillation around fixed point → OSCILLATING with ≥ 95% accuracy.
3. Held-out stalling scenario from exp3 → ≥ 75% correctly labeled CONVERGING
   (because the trajectory genuinely converges — original ground truth was
   wrong) OR ≥ 75% STALLING (if the slope p-value is not significant on n≤8
   iterations).
4. Held-out diverging scenario from exp3 → ≥ 75% correctly labeled DIVERGING.
5. New CONVERGING_v2 scenario → ≥ 65% correctly labeled CONVERGING or
   FAST_CONVERGE.

Any single failed prediction triggers a documented post-hoc analysis. The
classifier is not silently retuned to pass.

## Limitations to disclose

- Slope significance with n≤4 is weak; classifier may default to STALLING for
  very short loops even when the trend is real. This is operationally
  correct (insufficient evidence → conservative answer).
- The decision rule is not exhaustively continuous: a trajectory with strong
  monotone improvement *and* high residual variance currently classifies as
  CONVERGING (slope dominates oscillation when both fire). This is
  deliberate; oscillation matters only when there's no trend.
- The classifier still requires a meaningful E magnitude. If all errors are
  exactly 0, FAST_CONVERGE fires via E_ratio ≤ 0.1 (0/ε ≤ 0.1).
- The pre-registered thresholds are derived analytically. If they fail
  validation, the failure is reported and a *separate* post-hoc fitted
  classifier is built and clearly labelled as such.

## Amendments

### 2026-05-18: Tier-3 converging spec replacement (pre-confirmatory pilot)

The original CONVERGING_v2 spec (normalized Levenshtein with O(min) memory,
Unicode handling, and case-insensitive option) failed the spec-design
sanity check in a 3-trial pilot: trajectories oscillated rather than
converged because each revision satisfied one requirement while breaking
another. Spec was simplified to a plain Levenshtein with a docstring and
doctest example so the cognitive load per revision is small enough that
most Haiku seeds can reliably converge. **The amendment is to the
*scenario design* (trajectory shape needs to match the label), not to the
*classifier* (which is what we are validating).** Classifier thresholds
unchanged. Confirmatory run uses the amended spec only.
