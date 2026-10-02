# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settling-time / overshoot / steady-state-error metrics, adapted from classic step-response
control analysis to this project's *continuous* motion-tracking task.

The classic versions (see e.g. a PID step-response plot) assume the system is driven toward one
fixed setpoint and ask "how fast does it get there, does it overshoot, what error remains after it
settles". Our task has no single setpoint -- the reference pose moves every step -- so these only
make sense applied to the one portion of an eval run that *does* look like a step response: the
very start, where every env is simultaneously reset to a (randomized) pose and must converge onto
the moving reference for the first time. After that, envs re-reset asynchronously as individual
episodes end, so the per-step error_body_pos trace play.py already logs (mean across envs, by
construction) no longer has clean synchronized episode boundaries to key off of -- hence why these
metrics only look at the initial transient, not every reset throughout the run.

Pure Python, no Isaac Lab / torch dependency, so it's standalone-testable (see
scripts/windows/_test_eval_metrics.py in the humanoid-amass-kit repo).
"""

from __future__ import annotations

import statistics
from typing import Sequence


def compute_settling_metrics(
    error_series: Sequence[float],
    dt: float,
    threshold_ratio: float = 1.2,
    consecutive: int = 10,
    warmup_steps: int = 2,
) -> dict[str, float | bool | None]:
    """Settling time / overshoot ratio / steady-state error from a per-step error trace.

    Args:
        error_series: error_body_pos averaged across envs at each env.step() call, in time order,
            starting right after the synchronized initial reset (i.e. play.py's eval_log as-is).
        dt: seconds per step (``env.unwrapped.step_dt``), to convert the settling index to seconds.
        threshold_ratio: a step counts as "settled" once its error drops to within this multiple of
            the eventual steady-state level (see below) and stays there for ``consecutive`` steps.
        consecutive: how many steps in a row must be under threshold before declaring settled --
            guards against a single lucky low-error step triggering a false "settled" reading.
        warmup_steps: drop this many steps from the very start before computing anything.
            ``MotionCommand``'s relative-body-pose buffers are zero-initialized and only get their
            first real values once ``_update_command()`` runs, so the very first metrics read (taken
            right after the first ``env.step()``) measures against that stale zero buffer and comes
            out enormous -- a pure measurement artifact, not real tracking error, confirmed by every
            eval showing it as a ~50-90x "overshoot" with 1-step "settling". Dropping a couple of
            warmup steps removes it without meaningfully affecting anything else (default
            ``EVAL_STEPS``=1500, so 2 steps is 0.04s out of 30s).

    Returns:
        - ``settling_time_s``: seconds from the start (including warmup_steps) until settled, or
          ``None`` if it never settles within the run.
        - ``overshoot_ratio``: peak error before settling, divided by the steady-state level. 1.0
          would mean no overshoot at all; ``None`` if the steady-state level is ~0 (undefined ratio).
        - ``steady_state_error``: mean error after settling (or, if it never settles, the mean of
          the back half of the run -- the closest available approximation).
        - ``settled``: whether the run actually reached and held the threshold.
    """
    error_series = list(error_series)[warmup_steps:]
    n = len(error_series)
    if n == 0:
        return {"settling_time_s": None, "overshoot_ratio": None, "steady_state_error": None, "settled": False}

    half = max(1, n // 2)
    steady_state_level = statistics.median(error_series[-half:])
    threshold = steady_state_level * threshold_ratio

    settle_idx = None
    run = 0
    for i, e in enumerate(error_series):
        if e <= threshold:
            run += 1
            if run >= consecutive:
                settle_idx = i - consecutive + 1
                break
        else:
            run = 0
    settled = settle_idx is not None

    settling_time_s = (settle_idx + warmup_steps) * dt if settled else None
    overshoot_window = error_series[: settle_idx + consecutive] if settled else error_series
    overshoot_ratio = (max(overshoot_window) / steady_state_level) if steady_state_level > 1e-9 else None
    steady_state_error = statistics.mean(error_series[settle_idx:]) if settled else statistics.mean(error_series[half:])

    return {
        "settling_time_s": settling_time_s,
        "overshoot_ratio": overshoot_ratio,
        "steady_state_error": steady_state_error,
        "settled": settled,
    }
