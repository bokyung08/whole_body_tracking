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

4. ``performance_conditional_termination_threshold_curriculum`` (P, the proposed fix): an alternative to
   (2) that decays each termination threshold independently per motion clip, gated on that clip's own
   recent success-rate EMA rather than on the global step count. See its docstring for the diagnosed R2
   failure mode (fixed-schedule tightening disproportionately hurts fast/running clips) this targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

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
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # error_body_lin_vel/error_body_ang_vel aren't pre-allocated in MotionCommand.__init__ (only
    # error_anchor_*/error_body_pos/error_body_rot/error_joint_* are) -- they get added lazily on the
    # first metrics update, which happens AFTER the very first env.reset() that creates this curriculum
    # call. So the key may simply not exist yet; treat that the same as "no data this call" (env_ids
    # empty) rather than letting it crash iteration 0 of every run.
    metric = command.metrics.get(metric_name)
    if metric is None:
        return term_cfg.params["std"]
    error = metric[env_ids]
    if error.numel() == 0:
        return term_cfg.params["std"]
    batch_mean = error.mean().item()

    if not hasattr(env, "_amt_ema"):
        env._amt_ema = {}
    if term_name not in env._amt_ema:
        env._amt_ema[term_name] = batch_mean
    else:
        env._amt_ema[term_name] = (1.0 - ema_alpha) * env._amt_ema[term_name] + ema_alpha * batch_mean

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


def _update_clip_success_ema(env: ManagerBasedRLEnv, env_ids: Sequence[int], ema_alpha: float) -> torch.Tensor:
    """클립별 최근 성공률 EMA. 한 env.step()에서 anchor_pos/anchor_ori/ee_body_pos 세 커리큘럼
    term이 모두 이 함수를 부르더라도, ``env.common_step_counter`` 스탬프로 같은 스텝 안의 중복
    갱신을 막아 스텝당 정확히 한 번만 EMA에 반영한다 (CurriculumManager가 term들을 어떤 순서로
    부르든 상관없이 안전).

    "성공" = 이번에 끝난 에피소드가 anchor_pos/anchor_ori/ee_body_pos 중 하나로 조기종료되지
    않았다는 뜻 (``termination_manager.terminated`` -- time_out은 포함하지 않음, 즉 이 세
    DoneTerm 중 하나라도 걸리면 실패로 센다).
    """
    command = env.command_manager.get_term("motion")
    num_clips = command.motion.num_clips

    if not hasattr(env, "_p_clip_success_ema"):
        env._p_clip_success_ema = torch.ones(num_clips, device=env.device)
        env._p_ema_updated_step = -1

    if len(env_ids) > 0 and env.common_step_counter != env._p_ema_updated_step:
        clip_ids = command.motion_ids[env_ids]
        succeeded = (~env.termination_manager.terminated[env_ids]).float()
        sums = torch.zeros(num_clips, device=env.device).scatter_add_(0, clip_ids, succeeded)
        counts = torch.zeros(num_clips, device=env.device).scatter_add_(0, clip_ids, torch.ones_like(succeeded))
        has_data = counts > 0
        batch_rate = sums / counts.clamp(min=1)
        env._p_clip_success_ema = torch.where(
            has_data,
            (1.0 - ema_alpha) * env._p_clip_success_ema + ema_alpha * batch_rate,
            env._p_clip_success_ema,
        )
        env._p_ema_updated_step = env.common_step_counter

    return env._p_clip_success_ema


def performance_conditional_termination_threshold_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_key: str,
    init_value: float,
    min_value: float,
    step_size: float,
    success_ema_alpha: float = 0.05,
    tighten_above: float = 0.7,
    loosen_below: float = 0.4,
) -> float:
    """P (제안 방법): R2의 ``termination_threshold_curriculum``을 대체하는, 성능 조건부 버전.

    R2는 ``env.common_step_counter``만 보고 모든 클립에 똑같은 속도로 임계값을 조인다 --
    "빠른"(달리기 계열) 클립이든 "느린" 클립이든 같은 일정으로 조여지는데, 빠른 클립은 애초에
    추적 오차가 더 크기 때문에 더 일찍, 더 자주 조기종료에 걸려 연습 기회를 잃는다
    (진행상황_연구노트.md의 R2 클립별 실패 진단: success_rate 0.535->0.157, −37.9pp, 느린
    클립들은 오히려 개선됨). P는 전역 스텝 대신 "그 클립 자신의" 최근 성공률 EMA를 보고
    클립별로 독립적으로 조인다/푼다/유지한다 -- Rudin et al. (2021)의 지형 커리큘럼(승급/강급)
    아이디어를 종료 임계값에 적용한 것.

    성공률 EMA >= tighten_above: 그 클립의 임계값을 ``step_size``만큼 조인다 (min_value까지).
    성공률 EMA <  loosen_below : 그 클립의 임계값을 ``step_size``만큼 푼다 (init_value까지).
    그 사이                     : 그대로 유지.

    R2와 마찬가지로 AMT 리워드-std 커리큘럼/penalty 커리큘럼과는 독립적으로 켜고 끌 수 있다 --
    ``r2_curriculum.py``의 ``attach_p_curriculum``은 AMT std + penalty는 R2와 동일하게 두고
    이 함수만 바꿔 끼운다 (R2 대비 변경 변수를 하나로 좁혀, 효과를 정량적으로 분리 비교하기
    위함).
    """
    command = env.command_manager.get_term("motion")
    num_clips = command.motion.num_clips
    success_ema = _update_clip_success_ema(env, env_ids, success_ema_alpha)

    attr = f"_p_threshold_{term_name}"
    if not hasattr(env, attr):
        setattr(env, attr, torch.full((num_clips,), init_value, device=env.device))
    threshold = getattr(env, attr)

    tighten_mask = success_ema >= tighten_above
    loosen_mask = success_ema < loosen_below
    new_threshold = threshold.clone()
    new_threshold[tighten_mask] = (threshold[tighten_mask] - step_size).clamp(min=min_value)
    new_threshold[loosen_mask] = (threshold[loosen_mask] + step_size).clamp(max=init_value)
    setattr(env, attr, new_threshold)

    term_cfg = env.termination_manager.get_term_cfg(term_name)
    term_cfg.params[param_key] = new_threshold[command.motion_ids]
    env.termination_manager.set_term_cfg(term_name, term_cfg)
    return new_threshold.mean().item()


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
