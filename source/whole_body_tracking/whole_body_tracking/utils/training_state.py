"""Checkpoint curriculum progress and collect per-clip completed-episode diagnostics."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from types import MethodType

import torch


STATE_KEY = "tracking_training_state"
STATE_VERSION = 2


class TrackingTrainingState:
    def __init__(self, env, mode: str, log_dir: str):
        self.env = env
        self.mode = mode
        self.log_dir = Path(log_dir)
        self.command = env.command_manager.get_term("motion")
        files = self.command.cfg.motion_file
        files = [files] if isinstance(files, str) else list(files)
        self.clip_names = [Path(f).stem for f in files]
        if len(set(self.clip_names)) != len(self.clip_names):
            raise ValueError("Motion file names must be unique for per-clip diagnostics")
        self.signature = {
            "mode": mode,
            "clips": self.clip_names,
            "clip_lengths": self.command.motion.clip_len.cpu().tolist(),
            "num_envs": env.num_envs,
            "step_dt": env.step_dt,
            "curriculum": {
                name: {"function": cfg.func.__name__, "params": dict(cfg.params)}
                for name, cfg in vars(env.cfg.curriculum).items()
                if cfg is not None and hasattr(cfg, "func")
            },
        }
        self.reasons = list(env.termination_manager.active_terms)
        self.counts = {
            name: torch.zeros(len(files), dtype=torch.float64, device=env.device)
            for name in (
                "episodes", "successes", "failures", "length_sum", "length_squared_sum",
                "initial_excluded", "mixed_clip_excluded", "interrupted_excluded",
                "clip_end_reached", "clip_end_initial_excluded", "clip_end_mixed_excluded",
                "clip_end_length_sum", "clip_end_length_squared_sum",
                *(f"reason/{reason}" for reason in self.reasons),
            )
        }
        # A fresh simulator cannot restore unfinished physical episodes or the random length offset.
        self.initial_episode = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.mixed_clip_episode = torch.zeros_like(self.initial_episode)
        self.episode_clip = self.command.motion_ids.clone()
        self.in_reset = False
        self.resumed = False

    def attach(self):
        if hasattr(self.env, "_tracking_training_state"):
            raise RuntimeError("Tracking training state already attached")
        original_reset = self.env._reset_idx
        original_resample = self.command._resample_command
        state = self

        def reset_with_statistics(env, env_ids):
            state.record_reset(env_ids)
            state.in_reset = True
            try:
                return original_reset(env_ids)
            finally:
                state.in_reset = False
                state.mixed_clip_episode[env_ids] = False
                state.episode_clip[env_ids] = state.command.motion_ids[env_ids]

        def resample_with_statistics(command, env_ids):
            if not state.in_reset:
                state.record_clip_end(env_ids)
                state.mixed_clip_episode[env_ids] |= state.env.episode_length_buf[env_ids] > 0
            return original_resample(env_ids)

        self.env._reset_idx = MethodType(reset_with_statistics, self.env)
        self.command._resample_command = MethodType(resample_with_statistics, self.command)
        self.env._tracking_training_state = self
        return self

    def _add(self, key, clips, weights=None):
        values = torch.bincount(clips, weights=weights, minlength=len(self.clip_names))
        self.counts[key].add_(values)

    def record_clip_end(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        clips = self.command.motion_ids[ids]
        reached = (self.command.time_steps[ids] >= self.command.motion.clip_len[clips]) & (
            self.env.episode_length_buf[ids] > 0
        )
        done = self.env.termination_manager.terminated[ids] | self.env.termination_manager.time_outs[ids]
        ids, clips = ids[reached & ~done], clips[reached & ~done]
        initial = self.initial_episode[ids]
        mixed = self.mixed_clip_episode[ids] & ~initial
        self._add("clip_end_initial_excluded", clips[initial])
        self._add("clip_end_mixed_excluded", clips[mixed])
        # Close the first single-clip attempt before resampling; the later RL reset must not count it again.
        eligible = ~initial & ~mixed
        valid_clips = clips[eligible]
        lengths = self.env.episode_length_buf[ids[eligible]].to(torch.float64)
        self._add("clip_end_reached", valid_clips)
        self._add("clip_end_length_sum", valid_clips, lengths)
        self._add("clip_end_length_squared_sum", valid_clips, lengths.square())

    def record_reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        lengths = self.env.episode_length_buf[ids]
        active = lengths > 0
        if not active.any():
            return
        ids, lengths = ids[active], lengths[active]
        clips = self.command.motion_ids[ids]
        initial = self.initial_episode[ids]
        mixed = self.mixed_clip_episode[ids] & ~initial
        self._add("initial_excluded", self.episode_clip[ids[initial]])
        self._add("mixed_clip_excluded", self.episode_clip[ids[mixed]])
        self.initial_episode[ids] = False

        failed = self.env.termination_manager.terminated[ids]
        timeout = self.env.termination_manager.time_outs[ids]
        eligible = ~initial & ~mixed
        completed = eligible & (failed | timeout)
        self._add("interrupted_excluded", clips[eligible & ~completed])
        valid_clips = clips[completed]
        valid_lengths = lengths[completed].to(torch.float64)
        self._add("episodes", valid_clips)
        self._add("successes", clips[completed & timeout & ~failed])
        self._add("failures", clips[completed & failed])
        self._add("length_sum", valid_clips, valid_lengths)
        self._add("length_squared_sum", valid_clips, valid_lengths.square())
        for reason in self.reasons:
            mask = self.env.termination_manager.get_term(reason)[ids] & completed
            self._add(f"reason/{reason}", clips[mask])

    def capture(self):
        rewards = {}
        for name in self.env.reward_manager.active_terms:
            cfg = self.env.reward_manager.get_term_cfg(name)
            rewards[name] = {"weight": float(cfg.weight)}
            if "std" in cfg.params:
                rewards[name]["std"] = float(cfg.params["std"])
        thresholds = {}
        for name in self.reasons:
            cfg = self.env.termination_manager.get_term_cfg(name)
            value = cfg.params.get("threshold")
            if value is not None:
                if isinstance(value, torch.Tensor):
                    raise ValueError("Draft per-env P thresholds are not supported by this checkpoint format")
                thresholds[name] = float(value)
        return {
            "version": STATE_VERSION,
            "signature": deepcopy(self.signature),
            "common_step_counter": int(self.env.common_step_counter),
            "sim_step_counter": int(self.env._sim_step_counter),
            "amt_ema": dict(getattr(self.env, "_amt_ema", {})),
            "rewards": rewards,
            "thresholds": thresholds,
            "counts": {name: value.cpu().tolist() for name, value in self.counts.items()},
        }

    def restore(self, infos):
        saved = infos.get(STATE_KEY) if isinstance(infos, dict) else None
        if saved is None:
            if self.mode != "baseline":
                raise ValueError(
                    "This R2 checkpoint has no curriculum state; refusing an incorrect restart of "
                    "sigma/termination/penalty schedules. Use a new run name for a fresh run."
                )
            return
        if saved.get("version") != STATE_VERSION or saved.get("signature") != self.signature:
            raise ValueError("Tracking checkpoint version, curriculum, motion set, or environment settings differ")
        restored_counts = {}
        if set(saved["counts"]) != set(self.counts):
            raise ValueError("Tracking checkpoint statistics schema differs")
        for name, values in saved["counts"].items():
            value = torch.tensor(values, dtype=torch.float64, device=self.env.device)
            if value.shape != self.counts[name].shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid checkpoint statistics: {name}")
            restored_counts[name] = value
        for value in saved["amt_ema"].values():
            if not math.isfinite(value):
                raise ValueError("Non-finite checkpoint EMA")
        for name, values in saved["rewards"].items():
            cfg = self.env.reward_manager.get_term_cfg(name)
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite checkpoint reward: {name}")
            cfg.weight = values["weight"]
            if "std" in values:
                cfg.params["std"] = values["std"]
            self.env.reward_manager.set_term_cfg(name, cfg)
        for name, value in saved["thresholds"].items():
            if not math.isfinite(value):
                raise ValueError(f"Non-finite checkpoint threshold: {name}")
            cfg = self.env.termination_manager.get_term_cfg(name)
            cfg.params["threshold"] = value
            self.env.termination_manager.set_term_cfg(name, cfg)
        self.env.common_step_counter = int(saved["common_step_counter"])
        self.env._sim_step_counter = int(saved["sim_step_counter"])
        self.env._amt_ema = dict(saved["amt_ema"])
        self.counts = restored_counts
        self.resumed = True

    def write_snapshot(self, iteration):
        saved = self.capture()
        clips = {}
        for index, name in enumerate(self.clip_names):
            counts = {key: values[index] for key, values in saved["counts"].items()}
            n = counts["episodes"]
            mean = counts["length_sum"] / n if n else None
            variance = max(counts["length_squared_sum"] / n - mean**2, 0.0) if n else None
            attempts = n + counts["clip_end_reached"]
            successes = counts["successes"] + counts["clip_end_reached"]
            attempt_mean = (counts["length_sum"] + counts["clip_end_length_sum"]) / attempts if attempts else None
            attempt_variance = max(
                (counts["length_squared_sum"] + counts["clip_end_length_squared_sum"]) / attempts - attempt_mean**2,
                0.0,
            ) if attempts else None
            clips[name] = {
                **counts,
                "completed_attempts": attempts,
                "successful_attempts": successes,
                "success_rate": successes / attempts if attempts else None,
                "timeout_success_rate": counts["successes"] / n if n else None,
                "mean_episode_steps": mean,
                "std_episode_steps": math.sqrt(variance) if n else None,
                "mean_attempt_steps": attempt_mean,
                "std_attempt_steps": math.sqrt(attempt_variance) if attempts else None,
            }
        record = {
            "version": STATE_VERSION,
            "iteration": iteration,
            "common_step_counter": saved["common_step_counter"],
            "resumed": self.resumed,
            "mode": self.mode,
            "rewards": saved["rewards"],
            "thresholds": saved["thresholds"],
            "clips": clips,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / "clip_training_stats.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
