# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions implementing KungfuBot-style (Xie et al., 2025, arXiv:2506.12851) adaptive
motion tracking and reward/termination curricula for the DeepMimic-style tracking reward defined in
``tracking_env_cfg.RewardsCfg``/``TerminationsCfg``.

Three mechanisms, all opt-in (only attached to ``env_cfg.curriculum`` when the ``--r2_curriculum`` CLI
flag is set; the A/B/C/R1/D1 conditions leave ``CurriculumCfg`` empty as before and are unaffected):

1. ``adapt_tracking_std``: replaces a fixed ``std`` in one of the six exponential tracking-reward terms
   with an adaptively-shrinking one, following the closed-form update rule sigma <- min(sigma, EMA(error))
   derived in KungfuBot from a bi-level optimization of the tracking objective. Since it only shrinks, the
   tolerance monotonically tightens as the policy gets better at tracking, targeted at the observation
   (from the raw-vs-processed / reward-design analysis in this project) that some fixed stds -- notably
   the angular-velocity terms' std=3.14 -- are so loose that they carry almost no training signal.
2. ``termination_threshold_curriculum``: exponentially decays an early-termination distance/angle
   threshold from a lenient starting value toward a tighter floor, so exploration isn't crippled early on.
3. ``penalty_weight_curriculum``: exponentially grows a multiplier (0.1 -> 1.0 by default) applied on top
   of a regularization reward term's base weight (joint_limit / action_rate_l2 / undesired_contacts),
   so penalties bite harder only once the policy has something to lose.

Both (2) and (3) are step functions of ``env.common_step_counter`` (deterministic, resume-safe within a
single training phase) evaluated in closed form -- no persistent state needed. (1) needs an EMA, which is
stashed on the env object itself (``env._amt_ema``) since curriculum terms are plain functions with no
instance state of their own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def adapt_tracking_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    metric_name: str,
    ema_alpha: float = 0.01,
    min_std: float = 0.05,
) -> float:
    """KungfuBot-style adaptive tracking factor: sigma <- min(sigma, EMA(instantaneous error)).

    Called at env reset (env_ids = the environments that just terminated), so ``metric_name`` is read
    from ``MotionCommand.metrics`` before those buffers get overwritten by the reset -- i.e. it reflects
    the tracking error accumulated over the episode that just ended.

    Args:
        term_name: Name of the reward term in ``RewardsCfg`` whose ``params["std"]`` gets updated.
        metric_name: Name of the matching key in ``MotionCommand.metrics`` (e.g. "error_body_pos").
        ema_alpha: EMA smoothing factor for the error signal.
        min_std: Floor so std can't collapse to (near-)zero and starve the reward/gradient.
    """
    command = env.command_manager.get_term("motion")
    error = command.metrics[metric_name][env_ids]
    if error.numel() == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        return term_cfg.params["std"]
    batch_mean = error.mean().item()

    if not hasattr(env, "_amt_ema"):
        env._amt_ema = {}
    if term_name not in env._amt_ema:
        env._amt_ema[term_name] = batch_mean
    else:
        env._amt_ema[term_name] = (1.0 - ema_alpha) * env._amt_ema[term_name] + ema_alpha * batch_mean

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    new_std = min(term_cfg.params["std"], max(env._amt_ema[term_name], min_std))
    term_cfg.params["std"] = new_std
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return new_std


def termination_threshold_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_key: str,
    init_value: float,
    min_value: float,
    decay_per_step: float = 2.5e-5,
) -> float:
    """Exponentially decay a termination term's distance/angle threshold toward ``min_value``.

    Recomputed from ``env.common_step_counter`` in closed form each call (no persistent state), so it's
    well-defined regardless of how often/which env_ids trigger it.
    """
    new_value = max(init_value * (1.0 - decay_per_step) ** env.common_step_counter, min_value)
    term_cfg = env.termination_manager.get_term_cfg(term_name)
    term_cfg.params[param_key] = new_value
    env.termination_manager.set_term_cfg(term_name, term_cfg)
    return new_value


def penalty_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    base_weight: float,
    alpha_min: float = 0.1,
    alpha_max: float = 1.0,
    growth_per_step: float = 1.0e-4,
) -> float:
    """Exponentially grow a regularization reward term's effective weight from ``alpha_min *
    base_weight`` to ``alpha_max * base_weight`` (both assumed to share ``base_weight``'s sign).
    """
    alpha = min(alpha_min * (1.0 + growth_per_step) ** env.common_step_counter, alpha_max)
    new_weight = alpha * base_weight
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return new_weight
