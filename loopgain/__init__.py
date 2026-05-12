"""LoopGain — Barkhausen stability monitor for AI agent loops.

Public API:

    from loopgain import LoopGain
    guard = LoopGain(target_error=0.1)
    while guard.should_continue():
        guard.observe(errors, output=output)
    result = guard.result
"""

from loopgain.core import (
    LoopGain,
    LoopGainResult,
    ThresholdBands,
    INIT,
    FAST_CONVERGE,
    CONVERGING,
    STALLING,
    OSCILLATING,
    DIVERGING,
    TARGET_MET,
    MAX_ITERATIONS,
)

__version__ = "0.1.0"

__all__ = [
    "LoopGain",
    "LoopGainResult",
    "ThresholdBands",
    "INIT",
    "FAST_CONVERGE",
    "CONVERGING",
    "STALLING",
    "OSCILLATING",
    "DIVERGING",
    "TARGET_MET",
    "MAX_ITERATIONS",
    "__version__",
]
