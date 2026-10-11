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

R2's three sub-mechanisms (AMT reward-std reduction, termination-threshold reduction, penalty-weight
growth) can be mixed and matched to isolate which one drives which effect. This is motivated by the
diagnosed R2 fast-clip failure (진행상황_연구노트.md): R2 improves slower clips but severely degrades fast
(running/sprinting) clips -- success_rate 0.535 -> 0.157 on those clips specifically, while error_body_pos
on them actually worsens. Two competing explanations must be tested: the fixed termination/penalty
schedule and the initial reward-std collapse. TensorBoard records show the old screening run used
position std=0.045 and orientation std=0.06 from iteration 0 throughout; it did not test adaptive
position/orientation stds as intended. The startup-reset bug is now fixed, so R2-sigma is a corrected
adaptive condition, NOT an exact reproduction of the screening run's sigma component.

  attach_r2_curriculum        R2 (full): AMT std + fixed-schedule threshold + penalty.
  attach_r2_sigma_curriculum  R2-sigma (ablation): AMT std ONLY -- isolates the reward-shaping effect.
  attach_r2_theta_curriculum  R2-theta (ablation): fixed-schedule threshold + penalty ONLY, no AMT std --
                               isolates the termination/penalty-schedule effect (the suspected culprit).
  attach_p_curriculum         P (draft, NOT YET VALIDATED -- see note below): AMT std + a performance-
                               conditional threshold mechanism + penalty.

IMPORTANT correction (2026-10-11, after reading the actual papers rather than assuming): KungfuBot
(arXiv:2506.12851) Appendix C.5 was checked directly -- its termination-threshold curriculum (theta <-
clip(theta*(1-delta), theta_min, theta_max), delta=2.5e-5 in the paper) and its penalty curriculum are
BOTH pure step-count schedules with NO performance/success-rate gating whatsoever. So "R2 is KungfuBot's
own mechanism" is accurate, but citing KungfuBot as grounding for P's performance-conditional IDEA was
wrong -- KungfuBot is precisely the fixed-schedule design we are trying to improve on, not a precedent for
conditioning on performance. The two real precedents for that idea are:
  - Rudin et al. 2021 (CoRL; terrain curriculum in legged_gym's `_update_terrain_curriculum`): promotes to
    a harder terrain level when distance travelled this one episode > terrain_length/2; demotes when
    distance < 0.5 * commanded_distance. Single-episode binary decision, no averaging buffer.
  - OpenAI ADR (arXiv:1910.07113, Appendix C.3 Table 15): per-parameter performance buffer of length
    m=240 episodes; once full, average performance p_bar is compared against t_L=10/t_H=20 (out of up to
    50 per-episode successes); expand boundary by delta=0.02 if p_bar>=t_H, shrink if p_bar<=t_L, then
    clear the buffer. Buffered-average-then-step, not a continuous EMA.
`performance_conditional_termination_threshold_curriculum`'s current EMA + tighten_above=0.7/
loosen_below=0.4 implementation below was my own ad-hoc instantiation, not actually derived from either
paper -- user flagged this (2026-10-11) as insufficiently rigorous. Per-user decision: do NOT finalize or
run attach_p_curriculum yet. First run attach_r2_theta_curriculum and attach_r2_sigma_curriculum (with the
30_000-iteration-recalibrated constants below) and look at the results, THEN design P's exact mechanism
(likely restructured around the ADR buffer-and-step pattern) informed by those ablations AND a full
corrected R2 run. Neither single-component results nor a change relative to the old screening budget
establish that decay recalibration alone fixes the fast-clip problem.
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
# roughly step 432_000 of ~720_000 total (30_000 iters x 24 steps/iter, the 본실험 budget), i.e. ~60%
# into training -- same relative KungfuBot-style milestone (arXiv:2506.12851 Appendix C.5 uses the exact
# formula theta <- clip(theta*(1-decay), theta_min, theta_max)) as the project's earlier screening runs,
# just recalibrated for 30_000 iterations instead of the 8_000 this constant was originally derived for.
# 2026-10-11 재보정: 기존 상수(4.25e-6/3.92e-6)는 8,000 iteration(192,000 step) 기준으로 계산된 값이라, 그걸
# 그대로 30,000 iteration에 쓰면 전체의 ~17%(iteration 5,000)에서 이미 바닥에 도달하고 나머지 83%는 그대로
# 고정돼버림(의도한 "60% 지점까지 서서히" 와 다름) -- decay_per_step = -ln(min/init) / 432_000 로 재계산.
_TERMINATION_TERMS = [
    ("anchor_pos", 0.25, 0.15, 1.1824e-6),
    ("anchor_ori", 0.8, 0.5, 1.0880e-6),
    ("ee_body_pos", 0.25, 0.15, 1.1824e-6),
]

# Same (term_name, init_value, min_value) as above, but the 4th element is now a per-step-size for the
# performance-conditional version (P) instead of a decay rate -- chosen so a clip tightening every call
# reaches min_value in roughly the same number of calls as R2's fixed schedule would, for comparability.
_P_TERMINATION_TERMS = [
    ("anchor_pos", 0.25, 0.15, 5e-5),
    ("anchor_ori", 0.8, 0.5, 1.5e-4),
    ("ee_body_pos", 0.25, 0.15, 5e-5),
]

# (reward_term_name, base_weight) -- alpha ramps 0.1x -> 1.0x by ~step 432_000 (same 30_000-iteration
# recalibration as _TERMINATION_TERMS above; the old 1.921e-5 was derived for an 8_000-iteration budget).
_PENALTY_TERMS = [
    ("action_rate_l2", -1e-1),
    ("joint_limit", -10.0),
    ("undesired_contacts", -0.1),
]

_GROWTH_PER_STEP = 5.3305e-6


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
