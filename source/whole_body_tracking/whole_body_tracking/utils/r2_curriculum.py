# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attaches the R2 curriculum (KungfuBot-style adaptive motion tracking + termination/penalty
curricula, see ``tasks.tracking.mdp.curriculums``) onto an env cfg before ``gym.make``.

Opt-in only: called from ``scripts/rsl_rl/train.py`` when ``--r2_curriculum`` is passed. Every A/B/C/R1/D1
run leaves ``env_cfg.curriculum`` untouched (empty ``CurriculumCfg``), so this cannot affect their results.

The six tracking-reward stds and the three termination thresholds/penalty weights below are read directly
off ``tracking_env_cfg.RewardsCfg``/``TerminationsCfg`` (as of this writing: anchor_pos std=0.3, anchor_ori
std=0.4, body_pos std=0.3, body_ori std=0.4, body_lin_vel std=1.0, body_ang_vel std=3.14; anchor_pos
threshold=0.25, anchor_ori threshold=0.8, ee_body_pos threshold=0.25) so the curriculum starts exactly at
today's static values and only moves from there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

from whole_body_tracking.tasks.tracking.mdp.curriculums import (
    adapt_tracking_std,
    penalty_weight_curriculum,
    termination_threshold_curriculum,
)

# (reward_term_name, metric_name, init_std) -- min_std = 0.15 * init_std
_AMT_TERMS = [
    ("motion_global_anchor_pos", "error_anchor_pos", 0.3),
    ("motion_global_anchor_ori", "error_anchor_rot", 0.4),
    ("motion_body_pos", "error_body_pos", 0.3),
    ("motion_body_ori", "error_body_rot", 0.4),
    ("motion_body_lin_vel", "error_body_lin_vel", 1.0),
    ("motion_body_ang_vel", "error_body_ang_vel", 3.14),
]

# (termination_term_name, init_threshold, min_threshold, decay_per_step) -- reaches min_threshold at
# roughly step 120_000 of ~192_000 total (8000 iters x 24 steps/iter), i.e. ~60% into training.
_TERMINATION_TERMS = [
    ("anchor_pos", 0.25, 0.15, 4.25e-6),
    ("anchor_ori", 0.8, 0.5, 3.92e-6),
    ("ee_body_pos", 0.25, 0.15, 4.25e-6),
]

# (reward_term_name, base_weight) -- alpha ramps 0.1x -> 1.0x by ~step 120_000.
_PENALTY_TERMS = [
    ("action_rate_l2", -1e-1),
    ("joint_limit", -10.0),
    ("undesired_contacts", -0.1),
]

_GROWTH_PER_STEP = 1.921e-5


@configclass
class R2CurriculumCfg:
    """Dynamically populated below; kept as an empty configclass shell."""

    pass


def attach_r2_curriculum(env_cfg: "ManagerBasedRLEnvCfg") -> None:
    """Mutates ``env_cfg.curriculum`` in place, adding the AMT + termination + penalty curriculum terms."""
    curr_cfg = R2CurriculumCfg()

    for term_name, metric_name, init_std in _AMT_TERMS:
        setattr(
            curr_cfg,
            f"amt_{term_name}",
            CurrTerm(
                func=adapt_tracking_std,
                params={
                    "term_name": term_name,
                    "metric_name": metric_name,
                    "ema_alpha": 0.01,
                    "min_std": 0.15 * init_std,
                },
            ),
        )

    for term_name, init_value, min_value, decay in _TERMINATION_TERMS:
        setattr(
            curr_cfg,
            f"term_curr_{term_name}",
            CurrTerm(
                func=termination_threshold_curriculum,
                params={
                    "term_name": term_name,
                    "param_key": "threshold",
                    "init_value": init_value,
                    "min_value": min_value,
                    "decay_per_step": decay,
                },
            ),
        )

    for term_name, base_weight in _PENALTY_TERMS:
        setattr(
            curr_cfg,
            f"penalty_curr_{term_name}",
            CurrTerm(
                func=penalty_weight_curriculum,
                params={
                    "term_name": term_name,
                    "base_weight": base_weight,
                    "alpha_min": 0.1,
                    "alpha_max": 1.0,
                    "growth_per_step": _GROWTH_PER_STEP,
                },
            ),
        )

    env_cfg.curriculum = curr_cfg
    print("[INFO] R2 curriculum enabled (KungfuBot-style): adaptive tracking std + termination/penalty curricula")
