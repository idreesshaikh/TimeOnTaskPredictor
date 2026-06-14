"""
totvlm/scoring.py
=================
Structural difficulty score in the spirit of TaskSense (Yin et al., 2025).

Used in build_parquets.py to pre-compute tasksense_ms for each step.
NOT called at inference — the VLM learns to predict difficulty from pixels.

Formula per step:
    tasksense_ms = base(operation) + find_weight * log(n_interactive + 1) + motor

The log(n+1) term (Hick/Hyman-style) captures diminishing cost: going from
2→4 options hurts more than 40→42. The +1 keeps it defined when n=0.

Source: TaskSense Table 3 fitted base difficulties (ms).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ── Fitted base difficulties (ms) — TaskSense Table 3 ────────────────────────
BASE_MS: dict[str, float] = {
    "compute":          5120.0,   # mental calculation / comparison
    "decide_implicit":  1506.0,   # unstated choice the user must infer
    "create":           1422.0,   # composing new content (typing)
    "extract":          1416.0,   # pulling a specific value from the page
    "verify":            778.0,   # confirming something is correct
    "decide_explicit":   742.0,   # choosing among visible options
    "find":              563.0,   # locating a target (scales with element count)
    "recall":            447.0,   # remembering something from earlier
    "orient":              5.0,   # getting bearings on a fresh screen
}

MOTOR_MS: float = 859.0   # baseline motor cost even for trivial steps

# ── Action → cognitive operation ──────────────────────────────────────────────
# Revisit with the full WebChain action vocabulary.
# Document changes here — this mapping directly affects your training signal.
ACTION_TO_OP: dict[str, str] = {
    "click":      "decide_explicit",
    "dblclick":   "decide_explicit",
    "tap":        "decide_explicit",
    "select":     "decide_explicit",
    "choose":     "decide_explicit",
    "type":       "create",
    "input":      "create",
    "fill":       "create",
    "scroll":     "find",
    "swipe":      "find",
    "search":     "find",
    "hover":      "orient",
    "navigate":   "orient",
    "goto":       "orient",
    "launchapp":  "orient",
    "submit":     "verify",
    "drag":       "compute",
    "drop":       "compute",
}
DEFAULT_OP = "decide_explicit"


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class DifficultyBreakdown:
    """
    Score plus components — keep all parts for debugging.

    Log components (not just total) so you can see what the model
    is actually learning from during training.
    """
    operation:    str
    base_ms:      float
    find_term_ms: float   # find_weight * log(n + 1)
    motor_ms:     float
    total_ms:     float

    @property
    def score(self) -> float:
        """Difficulty in seconds (stored in parquet as tasksense_ms / 1000)."""
        return self.total_ms / 1000.0


# ── Main function ─────────────────────────────────────────────────────────────
def tasksense_difficulty(
    action_type:   str | None,
    n_interactive: int,
) -> DifficultyBreakdown:
    """
    Estimate structural difficulty for one step.

    Parameters
    ----------
    action_type
        Action the user took (click / type / scroll / navigate / …).
    n_interactive
        Number of interactive elements on screen (buttons, links, inputs, …).
        Comes from counting AX tree nodes with interactive roles.

    Returns
    -------
    DifficultyBreakdown
        .total_ms  : full score in milliseconds
        .score     : same, in seconds
        .operation : which cognitive operation was inferred
        .find_term_ms, .base_ms, .motor_ms : individual components
    """
    op         = ACTION_TO_OP.get((action_type or "").lower(), DEFAULT_OP)
    base       = BASE_MS[op]
    n          = max(0, int(n_interactive))
    find_term  = BASE_MS["find"] * math.log(n + 1)
    total      = base + find_term + MOTOR_MS

    return DifficultyBreakdown(
        operation    = op,
        base_ms      = base,
        find_term_ms = find_term,
        motor_ms     = MOTOR_MS,
        total_ms     = total,
    )
