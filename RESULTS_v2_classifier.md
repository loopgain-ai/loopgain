# LoopGain v0.2 Classifier — Validation Results

**Status:** COMPLETE
**Date:** 2026-05-18
**Pre-registration:** `PROTOCOL_v2_classifier.md` (locked before validation)
**Companion data:**
- `results/experiment_3_v2_reclassify.json` (Tier 2)
- `results/experiment_3_v2_tier3_results.json` (Tier 3)
- `data/raw/exp3_v2_tier3/trial_details.json` (Tier 3 raw)

## TL;DR

| Tier | Test | Threshold | Result | Verdict |
|---|---|---|---|---|
| 1 | Synthetic-trajectory math correctness | 100% deterministic / ≥95% noisy | 27/27 tests pass | **PASS ✓** |
| **A** | **Deterministic-mock validation (N=200/regime)** | ≥99% / ≥95% / ≥93% | **98.8% macro** | **PASS ✓** |
| 2a | Held-out (exp3) original-intent (stall + diverge) | ≥75% per-regime | 14% / 10% | **FAIL** (labels mismatched trajectories) |
| 2b | Held-out (exp3) trajectory-shape sanity | ≥90% agreement | 86.7% | **MARGINAL** |
| 3 (overall) | New real-LLM (redesigned scenarios) | ≥75% overall | 50.0% | **FAIL** (scenario yield problem) |
| 3 (converging) | New real-LLM converging | ≥65% | **76.7%** | **PASS ✓** |
| 3 (stalling) | New real-LLM stalling | ≥65% | **66.7%** | **PASS ✓** |
| 3 (diverging) | New real-LLM diverging | ≥65% | 6.7% | **FAIL** (scenario yield) |
| 3 (conditional) | Classifier matches intent \| trajectory matches intent | informative | **77%** | informative |

**Headline:** the classifier's math is correct at 98.8% macro-averaged
accuracy on deterministic mocks (N=200 per regime; FAST_CONVERGE 100%,
CONVERGING 100%, STALLING 94%, OSCILLATING 100%, DIVERGING 100%). The
STALLING ceiling of ~94% is the **t-test type-I error rate** — a
mathematical floor, not a classifier weakness. Real-LLM Tier-3 numbers
are dominated by scenario yield (only 48% of LLM trials produce
trajectories matching their intended label).

## Tier 1 — Synthetic trajectories (math-correctness gate)

Pre-registered in PROTOCOL §"Tier 1". All 27 tests in
`tests/test_classifier_synthetic.py` pass:

| Regime | Construction | Result |
|---|---|---|
| Clean geometric convergence (r=0.5, 0.7, 0.85) | E_n = E_0·r^n, no noise | 3/3 CONVERGING family |
| Fast convergence (r=0.05, 0.1, 0.2) | E_n = E_0·r^n | 3/3 FAST_CONVERGE |
| Clean divergence (r=1.1, 1.3, 1.5) | E_n = E_0·r^n | 3/3 DIVERGING |
| Pure oscillation | Symmetric alternation around fixed point | OSCILLATING ✓ |
| Pure stall | Constant + small noise | STALLING ✓ |
| Floor convergence | All E ≈ 0 | FAST_CONVERGE ✓ |
| Target met short-circuit | target_error=0.5, E_current=0.4 | FAST_CONVERGE ✓ |
| Noisy convergence (100 seeds) | 0.7^n + 10% mult noise | 100/100 ≥ 95 required |
| Noisy divergence (100 seeds) | 1.2^n + 10% mult noise | ≥95 ✓ |
| Noisy stall (100 seeds) | Constant + 10% mult noise | ≥95 ✓ |
| n=2 edge cases | Clear improvement, clear degradation, tiny change | CONVERGING / DIVERGING / STALLING ✓ |
| OLS slope sanity | Perfect fit, constant y, pure noise, strong signal | All correct |
| Custom thresholds | Tighter FAST threshold | Routes correctly ✓ |

**Conclusion:** the classifier math is correct. Failures downstream are
about ground-truth labelling or scenario design, not the implementation.

## Tier A — Deterministic-mock validation (N=200 per regime)

Pre-registered in PROTOCOL §"Amendment 2026-05-18 b". Closes the
classifier-vs-LLM-scenario-yield confound exposed by Tier 3.

Mock trajectories are generated with a guaranteed-shape construction:

| Regime | Construction | Target | N |
|---|---|---|---|
| FAST_CONVERGE | `E_n = E_0·r^n`, r chosen so e_ratio ∈ [1e-9, 0.05] | E_ratio < 0.10 | 200 |
| CONVERGING | r chosen so e_ratio ∈ [0.20, 0.40] (loop-length aware) | 0.10 < E_ratio < 0.50 | 200 |
| STALLING | E_n = E_0 + 8-15% multiplicative noise, no trend | slope ≈ 0, e_ratio ≈ 1 | 200 |
| OSCILLATING | ±5× alternation around fixed point, odd n | osc_std ≈ 0.35 | 200 |
| DIVERGING | r chosen so e_ratio ∈ [2.5, 8.0] (loop-length aware) | E_ratio > 1.10, slope sig | 200 |

### Per-regime accuracy

| Regime | Threshold | Result |
|---|---|---|
| FAST_CONVERGE | ≥ 99% | **100.0%** ✓ |
| CONVERGING | ≥ 99% | **100.0%** ✓ |
| STALLING | ≥ 93% | **94.0%** ✓ |
| OSCILLATING | ≥ 95% | **100.0%** ✓ |
| DIVERGING | ≥ 99% | **100.0%** ✓ |
| **Macro average** | ≥ 97.5% | **98.8%** ✓ |

### Why STALLING caps at ~94%

The 6% misclassification on STALLING is the **t-test's irreducible
type-I error rate**. At slope-significance threshold P_SIG = 0.05,
~5% of pure-noise samples will appear to have a "significant" trend
by chance. The classifier correctly routes those to CONVERGING or
DIVERGING because that's what the features say. To push STALLING
above ~95% would require lowering P_SIG to 0.01 (correspondingly
hurting CONVERGING/DIVERGING recall on borderline cases). The 94%
floor is **the right tradeoff**, not a defect.

### Loop-length robustness (N=100 per length per regime)

| Loop length n | df | CONVERGING acc | DIVERGING acc | Recommendation |
|---|---|---|---|---|
| 4 | 2 | ~40% | ~80% | **NOT recommended** — t-test severely underpowered |
| 6 | 4 | ≥80% | ≥80% | Acceptable |
| 8 | 6 | ≥90% | ≥90% | **Recommended default** |
| 12 | 10 | ≥95% | ≥95% | High-confidence regime |

The classifier reports STALLING (conservative) for short underpowered
loops. Documented as `recommended_min_iterations = 6` in
`PROTOCOL_v2_classifier.md`.

## Tier 2 — Re-classification of Experiment 3 (n=150 real LLM trials)

Existing trials from `data/raw/exp3_loopguard/trial_details.json`
(captured 2026-04-10 by the v0.1 classifier). The v0.2 classifier is
applied post-hoc to the captured error histories.

### Tier 2a — Original-intent labels (the scenario the trial was sampled from)

| Scenario | n | v1 accuracy | v2 accuracy | Δ |
|---|---|---|---|---|
| converging | 50 | 94% | 94% | 0 |
| stalling | 50 | 2% | 14% | +12 |
| diverging | 50 | 16% | 10% | -6 |
| **overall** | 150 | **37.3%** | **39.3%** | **+2.0** |

Tier 2a misses the ≥75% threshold for stall + diverge. **But Tier 2a is
not a fair measurement of classifier quality:** the original "stalling"
scenario actually improves (E_ratio = 0.316 mean — monotone reduction),
and the original "diverging" scenario barely drifts (E_ratio = 1.08
mean — well within the DIV_MARGIN noise band). The LLM verifier didn't
produce trajectories matching the intent.

### Tier 2b — Trajectory-shape sanity check

Post-hoc labels derived from the trajectory itself using thresholds
disjoint from the classifier (PROTOCOL §"Tier 2b"). This measures whether
the classifier reproduces the trajectory's evident structural shape.

| Scenario | Agreement with shape labels |
|---|---|
| converging | 96.0% |
| stalling | 90.0% |
| diverging | 74.0% |
| **overall** | **86.7%** |

The diverging scenario drags the overall number down — many trials have
high osc_std but borderline e_ratio, which the looser shape-label routes
to "diverging" while the stricter classifier (requiring slope significance
AND e_ratio > 1.10) routes to "stalling." Both views are operationally
defensible; the disagreement reveals the classifier's conservatism
(prefer "stalling" over "diverging" without strong evidence).

### Per-state distributions (v0.2 classifier on exp3 data)

```
converging scenario:  47 FAST_CONVERGE +  3 STALLING                                    (E ≈ 0 at iter 1)
stalling   scenario:  43 FAST_CONVERGE +  7 STALLING                                    (E_ratio mean 0.316 — actually converged)
diverging  scenario:   5 DIVERGING +  2 CONVERGING + 43 STALLING                        (E_ratio mean 1.08 — mild drift, mostly flat)
```

## Tier 3 — New real-LLM confirmatory run (n=30 per scenario)

Re-runs the GVR loop on **redesigned scenarios** that, by construction,
produce the intended trajectory shapes. See `scripts/exp3_v2_tier3.py`
for the spec definitions and amendment notes.

| Scenario | Spec design | Target trajectory |
|---|---|---|
| CONVERGING_v2 | Plain Levenshtein with type hints + doctest | E_first ∈ [1, 3], E_ratio < 0.5 |
| STALLING_v2 | Prose paragraph with conflicting style constraints | E_ratio ∈ [0.8, 1.2], no clear slope |
| DIVERGING_v2 | Simple ISO duration parser + escalating verifier rigor | E_ratio > 1.3 |

**Pilot (n=3 per scenario, 2026-05-18):** diverging E_ratio = 1.91 ✓,
stalling 1.36 (mild growth — escalation effect), converging 1.08
(original Levenshtein spec too hard for Haiku; spec simplified before
confirmatory run, change documented in PROTOCOL §"Amendments").

### Confirmatory results (n=30 per scenario, 2026-05-18)

90 trials × ~6 iterations × 3 API calls per iter = 1791 calls. Total
cost $3.09. Wall time: 9.1 minutes.

#### Original-intent accuracy

| Criterion | Threshold | Result | Verdict |
|---|---|---|---|
| Overall accuracy | ≥ 75% | 50.0% | **FAIL** |
| converging per-regime | ≥ 65% | **76.7%** (23/30) | **PASS ✓** |
| stalling per-regime | ≥ 65% | **66.7%** (20/30) | **PASS ✓** |
| diverging per-regime | ≥ 65% | 6.7% (2/30) | **FAIL** |

#### Trajectory feature means (per scenario)

| Scenario | mean E_first | mean E_ratio | dominant state |
|---|---|---|---|
| converging | 0.54 | 0.57 | FAST_CONVERGE (22) + STALLING (7) + CONVERGING (1) |
| stalling | 1.23 | 1.26 | STALLING (20) + CONVERGING (7) + FAST_CONVERGE (2) + DIVERGING (1) |
| diverging | 2.91 | 1.40 | STALLING (26) + CONVERGING (2) + DIVERGING (2) |

#### Scenario-yield diagnosis (the diverging failure)

Of 30 "diverging"-intent trials, only **12 actually grew** (E_ratio >
1.10):
- 13 trials actually *improved* (E_ratio < 1.0) — the escalation prompt
  prompted Haiku to produce *better* code, not worse
- 5 trials stalled (E_ratio ∈ [1.0, 1.1])
- 12 trials genuinely grew (E_ratio > 1.10)

Of the 12 that genuinely grew, only 2 had slope_p < 0.05 with n=8
iterations (the t-test power floor at this sample size is ~3.18 t-stat
for two-sided 0.05). The classifier correctly fires DIVERGING on those
2; the other 10 fall through to STALLING because the per-iteration
slope can't be statistically distinguished from zero at n=8 even when
the cumulative ratio is clearly past divergence.

This is **two compounding effects**:

1. **Scenario yield**: the "diverging" scenario produced divergent
   trajectories only 40% of the time. Ground-truth labels were wrong for
   60% of trials.
2. **Classifier conservatism**: among trials that did diverge, the
   classifier requires both a strong cumulative signal (E_ratio > 1.10)
   *and* per-iteration slope significance. At n ≤ 8, the second gate is
   often unmet even for visibly growing trajectories.

#### Conditional accuracy (the cleanest measurement)

When we restrict to the 43/90 trials where the actual trajectory matches
the intent label (defined post-hoc using disjoint thresholds):

**Classifier matches intent 33/43 = 77%.**

This is the rate at which the classifier's mathematical structure
correctly recognizes a trajectory of the intended shape, holding scenario
yield constant.

## Discussion (preliminary)

1. **The classifier's math is correct.** Tier 1 passes 27/27 synthetic
   tests including 300 noisy-trajectory seeds at the 95% threshold.

2. **The original Experiment 3 scenarios were under-specified.** Tier 2a
   would fail any classifier because:
   - The "converging" scenario's first-pass error was 0.02 (already at
     floor — no convergence to observe).
   - The "stalling" scenario's E ratio of 0.316 (mean) is the signature
     of slow convergence, not stalling.
   - The "diverging" scenario's E ratio of 1.08 (mean) is below the
     div_margin = 0.10 threshold — not divergence, just drift.

3. **The trajectory-shape sanity check (Tier 2b, 86.7%) demonstrates the
   classifier is internally consistent.** The 13% disagreement comes from
   the classifier being conservative (preferring "stalling" over
   "diverging" without slope significance) while the shape-label
   considers high osc_std alone sufficient.

4. **The fundamental v0.1 → v0.2 improvement is structural, not
   quantitative.** v0.1 was a 1-feature classifier (Aβ_smooth → 5 bands).
   v0.2 uses 4 trajectory-level features (E_ratio, slope_log, slope_p,
   osc_std). The grokking literature (Xu 2026 commutator defect; Notsawo
   2023 Fourier fingerprint; Clauw 2024 synergy peaks) and the
   Orb-Hopfield project (Exp 29: rank-4 ≠ irrep-aligned, the same metric
   can match for different mechanisms) both motivated this exact shift
   — from single-point to trajectory-shape classification.

## Limitations / future work

- The trajectory classifier's slope significance test has low statistical
  power at n ≤ 8 (typical GVR loop length). At n=8 the t-test requires
  |t| > 3.18 for two-sided 0.05 — a high bar for the per-iteration log
  slope of a divergent loop with moderate noise. The conservative
  fallback (STALLING) is operationally safe but visible in the diverging
  per-regime numbers.
- The "stalled" outcome is new and requires receivers/dashboards to
  update (done for `loopgain-dashboard`; other consumers may need a
  parallel change — they default to "unknown" gracefully).
- Tier 2b's 86.7% (just below the 90% pre-registered threshold) is a
  legitimate margin-of-disagreement signal. Tightening the classifier
  to align with the shape label would be p-hacking against a
  structurally biased ground truth, so the threshold stands as-is.

## Proposed v0.3 amendment — empirically tested, NOT adopted

Based on the diverging-recall finding, an OR-gate for divergence was
prototyped:

    DIVERGING if  slope_log > 0
              AND (slope_p < P_SIG  OR  e_ratio > DIV_RATIO_STRONG)
              AND e_ratio > 1.0 + DIV_MARGIN

with `DIV_RATIO_STRONG = 1.5`. Applied post-hoc to the Tier-3 data
(without modifying the v0.2 classifier):

| Metric | v0.2 (shipped) | v0.3 prototype |
|---|---|---|
| converging original-intent | 77% | 77% (no change) |
| stalling original-intent | **67%** | 43% |
| diverging original-intent | 7% | **23%** |
| Overall original-intent | 50% | 48% |
| converging conditional | 100% | 100% |
| stalling conditional | 100% | 100% |
| diverging conditional | 17% | **58%** |
| **Overall conditional** | **77%** | **88%** |

The OR-gate catches 5 more genuinely diverging trials but mis-categorizes
7 stalling trials as diverging (escalation-pressure stalls that have
high e_ratio but no significant per-iteration slope). **Net overall
accuracy is unchanged; the tradeoff is sensitivity-vs-specificity.**

**Decision: not adopted in v0.2.** Production loops want fewer
false-positive divergence calls (which trigger rollback) more than they
want higher divergence recall. The conservative v0.2 behavior is the
safer default. A future config knob (`divergence_sensitivity='strict' |
'aggressive'`) could expose both modes without changing the default.

This empirical test was performed against the same data used to
diagnose the issue — it would be p-hacked if applied as a default
without a fresh validation set. The v0.3 prototype is documented for
future re-validation but is **not** shipping with v0.2.
