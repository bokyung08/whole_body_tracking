# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attaches the R2 curriculum (KungfuBot-style adaptive motion tracking + termination/penalty
curricula, see ``tasks.tracking.mdp.curriculums``) onto an env cfg before ``gym.make`` -- and its
sigma/theta ablations plus the P (performance-conditional) proposed fix.

Opt-in only: called from ``scripts/rsl_rl/train.py`` when one of ``--r2_curriculum`` / ``--r2_sigma_only``
/ ``--r2_theta_only`` / ``--p_curriculum`` is passed. Every A/B/C/R1/D1 run leaves ``env_cfg.curriculum``
untouched (empty ``CurriculumCfg``), so this cannot affect their results.

The six tracking-reward stds and the three termination thresholds/penalty weights below are read directly
off ``tracking_env_cfg.RewardsCfg``/``TerminationsCfg`` (as of this writing: anchor_pos std=0.3, anchor_ori
std=0.4, body_pos std=0.3, body_ori std=0.4, body_lin_vel std=1.0, body_ang_vel std=3.14; anchor_pos
threshold=0.25, anchor_ori threshold=0.8, ee_body_pos threshold=0.25) so every curriculum below starts
exactly at today's static values and only moves from there.

R2's three sub-mechanisms (AMT reward-std tightening, termination-threshold tightening, penalty-weight
growth) can be mixed and matched to isolate which one drives which effect. This is motivated by the
diagnosed R2 fast-clip failure (진행상황_연구노트.md): R2 improves slower clips but severely degrades fast
(running/sprinting) clips -- success_rate 0.535 -> 0.157 on those clips specifically, while error_body_pos
on them actually worsens. The termination-threshold curriculum is the prime suspect, since it's the one
mechanism that directly causes early termination on a fixed, performance-blind schedule.

  attach_r2_curriculum        R2 (full): AMT std + fixed-schedule threshold + penalty.
  attach_r2_sigma_curriculum  R2-sigma (ablation): AMT std ONLY -- isolates the reward-shaping effect.
  attach_r2_theta_curriculum  R2-theta (ablation): fixed-schedule threshold + penalty ONLY, no AMT std --
                               isolates the termination/penalty-schedule effect (the suspected culprit).
  attach_p_curriculum         P (proposed fix): AMT std + performance-conditional threshold + penalty --
                               exactly R2 with ONLY the threshold mechanism swapped from a fixed schedule
                               to one gated on each motion clip's own recent success-rate EMA (see
                               ``performance_conditional_termination_threshold_curriculum``), so the
                               comparison against full R2 isolates that one change.
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
    performance_conditional_termination_threshold_curriculum,
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

# Same (term_name, init_value, min_value) as above, but the 4th element is now a per-step-size for the
# performance-conditional version (P) instead of a decay rate -- chosen so a clip tightening every call
# reaches min_value in roughly the same number of calls as R2's fixed schedule would, for comparability.
_P_TERMINATION_TERMS = [
    ("anchor_pos", 0.25, 0.15, 5e-5),
    ("anchor_ori", 0.8, 0.5, 1.5e-4),
    ("ee_body_pos", 0.25, 0.15, 5e-5),
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


def _add_amt_terms(curr_cfg: "R2CurriculumCfg") -> None:
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


def _add_fixed_schedule_termination_terms(curr_cfg: "R2CurriculumCfg") -> None:
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


def _add_performance_conditional_termination_terms(curr_cfg: "R2CurriculumCfg") -> None:
    for term_name, init_value, min_value, step_size in _P_TERMINATION_TERMS:
        setattr(
            curr_cfg,
            f"term_curr_{term_name}",
            CurrTerm(
                func=performance_conditional_termination_threshold_curriculum,
                params={
                    "term_name": term_name,
                    "param_key": "threshold",
                    "init_value": init_value,
                    "min_value": min_value,
                    "step_size": step_size,
                    "success_ema_alpha": 0.05,
                    "tighten_above": 0.7,
                    "loosen_below": 0.4,
                },
            ),
        )


def _add_penalty_terms(curr_cfg: "R2CurriculumCfg") -> None:
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


def attach_r2_curriculum(env_cfg: "ManagerBasedRLEnvCfg") -> None:
    """R2 (full): AMT std + fixed-schedule termination threshold + penalty-weight curricula."""
    curr_cfg = R2CurriculumCfg()
    _add_amt_terms(curr_cfg)
    _add_fixed_schedule_termination_terms(curr_cfg)
    _add_penalty_terms(curr_cfg)
    env_cfg.curriculum = curr_cfg
    print("[INFO] R2 curriculum enabled (KungfuBot-style): adaptive tracking std + termination/penalty curricula")


def attach_r2_sigma_curriculum(env_cfg: "ManagerBasedRLEnvCfg") -> None:
    """R2-sigma (ablation): AMT reward-std tightening ONLY, no threshold/penalty change."""
    curr_cfg = R2CurriculumCfg()
    _add_amt_terms(curr_cfg)
    env_cfg.curriculum = curr_cfg
    print("[INFO] R2-sigma curriculum enabled (ablation): adaptive tracking std ONLY")


def attach_r2_theta_curriculum(env_cfg: "ManagerBasedRLEnvCfg") -> None:
    """R2-theta (ablation): fixed-schedule termination threshold + penalty ONLY, no AMT std change."""
    curr_cfg = R2CurriculumCfg()
    _add_fixed_schedule_termination_terms(curr_cfg)
    _add_penalty_terms(curr_cfg)
    env_cfg.curriculum = curr_cfg
    print("[INFO] R2-theta curriculum enabled (ablation): fixed-schedule termination/penalty curricula ONLY")


def attach_p_curriculum(env_cfg: "ManagerBasedRLEnvCfg") -> None:
    """P (proposed fix): AMT std + performance-conditional termination threshold + penalty.

    Identical to attach_r2_curriculum except the termination-threshold mechanism is swapped from a fixed
    global-step schedule to one gated on each motion clip's own recent success-rate EMA (see
    performance_conditional_termination_threshold_curriculum) -- the single variable changed relative to
    full R2, so their results are directly comparable.
    """
    curr_cfg = R2CurriculumCfg()
    _add_amt_terms(curr_cfg)
    _add_performance_conditional_termination_terms(curr_cfg)
    _add_penalty_terms(curr_cfg)
    env_cfg.curriculum = curr_cfg
    print("[INFO] P curriculum enabled (proposed fix): adaptive tracking std + performance-conditional termination threshold + penalty")
