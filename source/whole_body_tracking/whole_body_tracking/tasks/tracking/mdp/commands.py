from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Loads one or more motion .npz files.

    Pass a single path for the original single-motion behavior (e.g. the ``smoke`` test
    that pulls one clip from the WandB registry). Pass a list of paths (e.g. all the npz
    files under a ``--motion_dir``) to concatenate them into one buffer and track each
    clip's offset/length via :attr:`clip_start`/:attr:`clip_len`, so :class:`MotionCommand`
    can sample across clips. All clips are assumed to share the same fps (csv_to_npz.py's
    default output is 50fps for all of them).
    """

    def __init__(self, motion_file: str | Sequence[str], body_indexes: Sequence[int], device: str = "cpu"):
        motion_files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        assert len(motion_files) > 0, "motion_file must be a path or a non-empty list of paths"
        for f in motion_files:
            assert os.path.isfile(f), f"Invalid file path: {f}"
        datas = [np.load(f) for f in motion_files]

        def cat(key: str) -> torch.Tensor:
            return torch.cat(
                [torch.tensor(d[key], dtype=torch.float32, device=device) for d in datas], dim=0
            )

        self.fps = datas[0]["fps"]
        self.joint_pos = cat("joint_pos")
        self.joint_vel = cat("joint_vel")
        self._body_pos_w = cat("body_pos_w")
        self._body_quat_w = cat("body_quat_w")
        self._body_lin_vel_w = cat("body_lin_vel_w")
        self._body_ang_vel_w = cat("body_ang_vel_w")
        self._body_indexes = body_indexes

        clip_lens = torch.tensor([d["joint_pos"].shape[0] for d in datas], dtype=torch.long, device=device)
        self.clip_len = clip_lens
        self.clip_start = torch.cumsum(clip_lens, dim=0) - clip_lens
        self.num_clips = len(datas)
        # kept for backward compatibility (single-clip code paths / replay_npz.py / exporter.py);
        # equals clip_len[0] when num_clips == 1, and the combined length across all clips otherwise.
        self.time_step_total = int(clip_lens.sum().item())

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # which clip (into self.motion's concatenated buffer) each env is currently following.
        # stays all-zero when there's only one clip, so single-motion behavior is unaffected.
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # the failure-rate-adaptive time-bin sampling below only makes sense within a single
        # motion; with multiple clips we sample clips uniformly instead (see _resample_command).
        # Keep bin_count trivial in that case rather than sizing it to the combined length of
        # every clip.
        if self.motion.num_clips == 1:
            self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        else:
            self.bin_count = 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # D2 (opt-in, cfg.adaptive_clip_sampling): same EMA-of-failure-rate idea as the time-bin
        # adaptive sampler above (_adaptive_sampling), but keyed by clip index instead of time-bin
        # index -- oversamples clips the policy currently fails on more often, instead of the static
        # "drop the always-failing clips" curation D1 did. See _adaptive_clip_sampling().
        self.clip_failed_count = torch.zeros(self.motion.num_clips, dtype=torch.float, device=self.device)
        self._current_clip_failed = torch.zeros(self.motion.num_clips, dtype=torch.float, device=self.device)

        # N1 (opt-in, cfg.segment_adaptive_sampling): 위 시간빈(time-bin) EMA 실패 샘플러를
        # 클립별로 독립적으로 복제한 버전. 자세한 내용은 _segment_adaptive_clip_sampling() 참고.
        self.segment_failed_count = torch.zeros(
            self.motion.num_clips, self.cfg.segment_bins, dtype=torch.float, device=self.device
        )
        self._current_segment_failed = torch.zeros(
            self.motion.num_clips, self.cfg.segment_bins, dtype=torch.float, device=self.device
        )

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _frame_idx(self) -> torch.Tensor:
        """Index into self.motion's concatenated-clips buffer for each env's current frame."""
        return self.motion.clip_start[self.motion_ids] + self.time_steps

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self._frame_idx()]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self._frame_idx()]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._frame_idx()] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._frame_idx()]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._frame_idx()]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._frame_idx()]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._frame_idx(), self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._frame_idx(), self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._frame_idx(), self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._frame_idx(), self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_clip_sampling(self, env_ids: Sequence[int]):
        """Multi-clip resampling: pick a clip uniformly, then a start frame uniformly within it.

        This is the "처음엔 클립 균등 샘플링 + 클립 안 시작 시점 균등 샘플링으로 단순화" simplification
        from docs/실험실행튜토리얼_baselineAB비교.md 6단계 — the failure-rate-adaptive bin sampling in
        _adaptive_sampling() only makes sense within a single clip's timeline, so with multiple
        clips we skip it rather than adapt it (candidate Stage-2 improvement: per-clip adaptive
        sampling). sampling_entropy/top1_prob/top1_bin metrics are left at their initial value
        (0) in this mode since they describe the (unused) bin distribution.
        """
        self.motion_ids[env_ids] = torch.randint(0, self.motion.num_clips, (len(env_ids),), device=self.device)
        clip_lens = self.motion.clip_len[self.motion_ids[env_ids]]
        self.time_steps[env_ids] = (
            sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * (clip_lens - 1).clamp(min=0)
        ).long()

    def _adaptive_clip_sampling(self, env_ids: Sequence[int]):
        """D2: pick a clip with probability proportional to its recent (EMA'd) failure rate,
        then a start frame uniformly within it. Mirrors _adaptive_sampling()'s bin_failed_count/
        adaptive_alpha/adaptive_uniform_ratio machinery exactly, but keyed by clip index -- no
        kernel smoothing (clip index order is arbitrary/alphabetical, unlike time bins, so
        smoothing across neighboring indices wouldn't mean anything)."""
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            fail_clips = self.motion_ids[env_ids][episode_failed]
            self._current_clip_failed[:] = torch.bincount(fail_clips, minlength=self.motion.num_clips)

        sampling_probabilities = self.clip_failed_count + self.cfg.adaptive_uniform_ratio / float(
            self.motion.num_clips
        )
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        self.motion_ids[env_ids] = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        clip_lens = self.motion.clip_len[self.motion_ids[env_ids]]
        self.time_steps[env_ids] = (
            sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * (clip_lens - 1).clamp(min=0)
        ).long()

        # Metrics (reused names: now describe the clip distribution rather than the time-bin one)
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.motion.num_clips)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.motion.num_clips

    def _segment_adaptive_clip_sampling(self, env_ids: Sequence[int]):
        """N1 (Stubborn 스타일, arXiv:2606.12814): 클립을 "고르는" 것 자체는 균등(uniform)하게
        유지한다(_uniform_clip_sampling 참고) — D2와 달리, 어떤 클립이 아무리 자주 실패해도
        전체 배치에서 그 클립이 차지하는 몫은 절대 1/num_clips를 넘을 수 없다. 대신 "고른 클립
        안에서 어디부터 시작할지"만, 그 클립 자신의 최근 실패 세그먼트 쪽으로 편향시킨다
        (cfg.segment_bins개 구간에 대한 EMA 실패율 히스토그램을 클립마다 따로 추적). 이는
        D2의 실패 원인(docs/진행상황_연구노트.md 2026-09-27)을 정면으로 겨냥한 수정이다: D2는
        클립 전체를 실패율로 재가중했기 때문에, 물리적으로 불가능한 클립 하나의 확률이
        포화되어(테스트에서 80~93%) 다른 모든 클립의 연습 기회를 빼앗았다."""
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            old_motion_ids = self.motion_ids[env_ids][episode_failed]
            old_clip_lens = self.motion.clip_len[old_motion_ids]
            old_segment = torch.clamp(
                (self.time_steps[env_ids][episode_failed] * self.cfg.segment_bins) // old_clip_lens.clamp(min=1),
                0,
                self.cfg.segment_bins - 1,
            )
            flat_idx = old_motion_ids * self.cfg.segment_bins + old_segment
            self._current_segment_failed[:] = torch.bincount(
                flat_idx, minlength=self.motion.num_clips * self.cfg.segment_bins
            ).view(self.motion.num_clips, self.cfg.segment_bins).float()

        # 클립은 균등하게 고른다 — 아래의 세그먼트 편향이 아니라 바로 "이 균등 선택"이 D2의
        # 실패 원인을 고치는 핵심이다.
        self.motion_ids[env_ids] = torch.randint(0, self.motion.num_clips, (len(env_ids),), device=self.device)
        clip_lens = self.motion.clip_len[self.motion_ids[env_ids]]

        # 고른 클립 "안에서는" 시작 세그먼트를 그 클립 자신의 최근 실패 이력 쪽으로 편향시킨다.
        probs = self.segment_failed_count[self.motion_ids[env_ids]] + self.cfg.adaptive_uniform_ratio / float(
            self.cfg.segment_bins
        )
        probs = probs / probs.sum(dim=-1, keepdim=True)
        sampled_segment = torch.multinomial(probs, 1).squeeze(-1)
        self.time_steps[env_ids] = (
            (sampled_segment + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.cfg.segment_bins
            * (clip_lens - 1).clamp(min=0)
        ).long()

        # 지표(기존 이름 재사용): 이번에 고른 클립 자신의 세그먼트 분포에 대한 env별 값 —
        # _adaptive_sampling이 전역 하나로 구했던 entropy/top1_prob/top1_bin을 env 배치별로
        # 구한 버전이라고 보면 된다.
        H = -(probs * (probs + 1e-12).log()).sum(dim=-1) / math.log(self.cfg.segment_bins)
        pmax, imax = probs.max(dim=-1)
        self.metrics["sampling_entropy"][env_ids] = H
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.cfg.segment_bins

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.motion.num_clips > 1:
            if self.cfg.adaptive_clip_sampling:
                self._adaptive_clip_sampling(env_ids)
            elif self.cfg.segment_adaptive_sampling:
                self._segment_adaptive_clip_sampling(env_ids)
            else:
                self._uniform_clip_sampling(env_ids)
        else:
            self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        # per-env clip length (clip_len[motion_ids] == time_step_total for every env when there's
        # only one clip, so this is equivalent to the original `>= self.motion.time_step_total`)
        env_ids = torch.where(self.time_steps >= self.motion.clip_len[self.motion_ids])[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        if self.cfg.adaptive_clip_sampling:
            self.clip_failed_count = (
                self.cfg.adaptive_alpha * self._current_clip_failed
                + (1 - self.cfg.adaptive_alpha) * self.clip_failed_count
            )
            self._current_clip_failed.zero_()

        if self.cfg.segment_adaptive_sampling:
            self.segment_failed_count = (
                self.cfg.adaptive_alpha * self._current_segment_failed
                + (1 - self.cfg.adaptive_alpha) * self.segment_failed_count
            )
            self._current_segment_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(self.cfg.show_body_markers)
                self.goal_body_visualizers[i].set_visibility(self.cfg.show_body_markers)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        if self.cfg.show_body_markers:
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
                self.goal_body_visualizers[i].visualize(
                    self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
                )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | list[str] = MISSING
    """A single motion .npz path, or a list of paths to train on multiple clips at once
    (each env samples a random clip + start frame; see MotionLoader/MotionCommand)."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    adaptive_clip_sampling: bool = False
    """D2 (opt-in, off by default): with multiple clips (--motion_dir), sample clips proportional
    to their recent EMA'd failure rate instead of uniformly. See MotionCommand._adaptive_clip_sampling."""

    segment_adaptive_sampling: bool = False
    """N1 (opt-in, 기본 꺼짐; Stubborn 스타일, arXiv:2606.12814): 클립이 여러 개일 때, 클립을
    "고르는" 것 자체는(D2와 달리) 균등하게 유지하되, 일단 고른 클립 "안에서는" 시작 프레임을
    균등 샘플링 대신 그 클립 자신의 최근 실패 세그먼트 쪽으로 편향시킨다. 이는 D2의 실제 실패
    원인을 직접 겨냥한다: D2는 클립 전체를 실패율로 재가중하기 때문에, 물리적으로 불가능한
    클립 하나의 확률이 포화되어 샘플링 예산 대부분을 집어삼킨다(docs/진행상황_연구노트.md
    2026-09-27). "어느 클립을 고를지"는 균등하게 두고 "그 클립 안 어디서 시작할지"만 편향시키면,
    풀 수 없는 클립이라도 절대 1/num_clips 몫 이상을 차지할 수 없다 — 그 몫을 자기 안에서
    낭비할 뿐, 다른 클립들의 몫을 빼앗지는 못한다. 자세한 구현은
    MotionCommand._segment_adaptive_clip_sampling 참고. adaptive_clip_sampling(D2)과는 동시에
    켤 수 없는 관계라, 둘 다 설정되면 D2가 우선한다(_resample_command 참고)."""

    segment_bins: int = 10
    """N1의 실패율 EMA를 추적하는 클립당 시간 구간 수(단일 클립용 _adaptive_sampling의
    bin_count보다 훨씬 성기다 — 그쪽은 제어 스텝 하나당 구간 하나 정도로 잡는데, 여기서 그렇게
    하면 `num_clips * 수백` 개의 EMA 칸이 생겨버려 낭비이기도 하고, 클립 하나당 받는 에피소드
    수가 단일 클립 때보다 훨씬 적어서 칸 하나하나의 값이 더 들쭉날쭉해진다)."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    show_body_markers: bool = True
    """Opt-out (default on, unchanged behavior): with ~14 tracked bodies, the full current+goal
    frame-marker set is dense enough to obscure the robot in rendered video. Set False (e.g. from
    play.py for a demo render) to show only the anchor (root) current/goal markers -- still proves
    the policy is tracking the reference, with far less visual clutter."""
