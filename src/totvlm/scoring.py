"""
scoring.py
==========
Turns a screen's accessibility (AX) tree into a single structural "difficulty"
number, in the spirit of TaskSense (Yin et al., 2025).

Where this fits in your research
--------------------------------
Your novelty is predicting Time-on-Task from *pixels only* at inference. But at
*training* time you are allowed to use richer signals to teach the model. The
TaskSense score is exactly that: a cheap, theory-grounded difficulty estimate
computed from the AX tree, used as auxiliary supervision during training and
then thrown away at test time (see schema.Sample: it is a "train-only" field).

The numbers below are the fitted base difficulties from TaskSense, Table 3
(milliseconds). Treat this file as a *first, honest approximation* you will
refine — it is deliberately small and readable so you can reason about it.

IMPORTANT (be a good scientist): the action -> cognitive-operation mapping here
is a simplification. Document any change you make, because this score feeds your
training signal and therefore your results. When in doubt, log the components,
not just the total, so you can debug what the model is actually learning from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Fitted base difficulty per cognitive operation, in milliseconds.
# Source: TaskSense (Yin et al., 2025), Table 3. These are the same values
# behind the bar chart on slide 8 of your proposal deck.
BASE_DIFFICULTY_MS: dict[str, float] = {
    "compute": 5120.0,  # mental calculation / comparison
    "decide_implicit": 1506.0,  # an unstated choice the user must infer
    "create": 1422.0,  # composing new content
    "extract": 1416.0,  # pulling a specific value out of the page
    "verify": 778.0,  # confirming something is correct
    "decide_explicit": 742.0,  # choosing among visible options
    "find": 563.0,  # locating a target element (scales with element count)
    "recall": 447.0,  # remembering something from earlier
    "orient": 5.0,  # getting one's bearings on a fresh screen
}

# Motor/intercept term: the baseline time even a trivial step costs.
MOTOR_INTERCEPT_MS: float = 859.0

# A coarse map from the logged action type to a cognitive operation.
# You WILL want to revisit this with the WebChain action vocabulary in hand.
ACTION_TO_OPERATION: dict[str, str] = {
    "click": "decide_explicit",
    "select": "decide_explicit",
    "type": "create",
    "input": "create",
    "scroll": "find",
    "search": "find",
    "hover": "orient",
    "navigate": "orient",
    "submit": "verify",
}

DEFAULT_OPERATION = "decide_explicit"  # used when an action type is unknown


@dataclass
class DifficultyBreakdown:
    """The score plus its parts — keep the parts so you can debug the signal."""

    operation: str
    base_ms: float
    find_term_ms: float
    motor_ms: float
    total_ms: float

    @property
    def score(self) -> float:
        """The structural difficulty in *seconds* (what we store on the Sample)."""
        return self.total_ms / 1000.0


def tasksense_difficulty(
    action_type: str | None,
    n_interactive: int,
) -> DifficultyBreakdown:
    """Estimate structural difficulty for one screen.

    Parameters
    ----------
    action_type
        The action the user took on this screen (from the AX/action log).
    n_interactive
        Number of interactive elements on the screen (buttons, links, inputs).
        This is the `n` in TaskSense's Find term: difficulty of *locating* a
        target grows with how many candidates there are.

    Returns
    -------
    DifficultyBreakdown
        The total and its components. Call `.score` for difficulty in seconds.

    The model
    ---------
        difficulty = base(operation) + find_weight * log(n_interactive + 1) + motor

    The `log(n + 1)` form (Hick/Hyman-style) captures diminishing cost: going
    from 2 to 4 options hurts more than going from 40 to 42. The `+1` keeps the
    term defined (and zero) when there are no interactive elements.
    """
    op = ACTION_TO_OPERATION.get((action_type or "").lower(), DEFAULT_OPERATION)
    base = BASE_DIFFICULTY_MS[op]

    # The "find" base difficulty doubles as the per-element search weight.
    find_weight = BASE_DIFFICULTY_MS["find"]
    n = max(0, int(n_interactive))
    find_term = find_weight * math.log(n + 1)

    total = base + find_term + MOTOR_INTERCEPT_MS
    return DifficultyBreakdown(
        operation=op,
        base_ms=base,
        find_term_ms=find_term,
        motor_ms=MOTOR_INTERCEPT_MS,
        total_ms=total,
    )
