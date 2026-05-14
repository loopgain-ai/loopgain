"""LoopGain — Barkhausen stability monitor for AI agent loops.

Public API:

    from loopgain import LoopGain
    lg = LoopGain(target_error=0.1)
    while lg.should_continue():
        lg.observe(errors, output=output)
    result = lg.result
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
from loopgain.telemetry import build_payload as build_telemetry_payload

__version__ = "0.1.5"

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
    "build_telemetry_payload",
    "__version__",
]
