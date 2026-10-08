"""N4 candidate (SA2RT-style, arXiv:2507.08303): a learned, state-conditioned, *selective*
adversarial perturbation policy, replacing the task's static uniform-random push
(`events.py`'s `push_by_setting_velocity` / `tracking_env_cfg.py`'s `push_robot` event, interval
1-3s, fixed `VELOCITY_RANGE`). Static domain randomization pushes every env by the same random
amount regardless of whether that push is actually destabilizing; the idea here is a small policy
that learns *where/how hard* to push so as to actually disrupt tracking, and that can learn to
push gently (or near not at all) in states where a hard push wouldn't teach the main policy
anything -- hence "selective".

Scope note (plan document's N4 is this single piece, not APEX-style full bi-level RL): this is a
simplified, environment-event-level implementation chosen specifically to avoid the much larger
and riskier change a true alternating actor-vs-actor training loop would need (forking
MotionOnPolicyRunner.learn() itself, a second rollout buffer, etc.). It still satisfies the
candidate's actual claim -- a *learned* perturber that adapts via its own reward signal, not a
fixed distribution -- via a simple contextual-bandit / one-step policy-gradient (REINFORCE)
update, trained online, in alternation with (but not blocking) the main PPO policy's own training.
See whole_body_tracking/utils/adversarial_perturber.py (pure torch, no isaaclab deps, unit-tested
standalone) for the perturber policy and its REINFORCE update; this file is just the thin
environment-event wrapper around it.

Each time this event fires for a given env (IsaacLab's interval-mode EventManager, same cadence as
the static push, 1-3s apart), it does two things:
  (1) *finishes* the previous push's credit assignment for that env, if there was one: the reward
      for that past action is this env's current motion_anchor_pos tracking error (i.e. "how
      disrupted is tracking, now that enough time has passed for the push's effect to show").
  (2) samples and applies a *new* push from the current (post-update) perturber policy, and
      remembers it as "pending" for the next call's credit assignment.

Not used unless `--adversarial_push` is passed (train.py sets env_cfg.events.push_robot.func to
this module's `adversarial_push`, same `params={"velocity_range": ...}` call signature as the
static `push_by_setting_velocity` it replaces, so no env_cfg structural change is needed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.utils.adversarial_perturber import AdversarialPerturber

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def adversarial_push(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Drop-in replacement for isaaclab.envs.mdp.push_by_setting_velocity with the same call
    signature (so it can be swapped in via a single `env_cfg.events.push_robot.func =` assignment)
    -- except the push is chosen by a small learned, selective adversarial policy instead of
    sampled uniformly at random. See module docstring."""
    asset: "RigidObject | Articulation" = env.scene[asset_cfg.name]

    perturber: AdversarialPerturber | None = getattr(env, "_sa2rt_perturber", None)
    if perturber is None:
        perturber = AdversarialPerturber(env.num_envs, env.device)
        env._sa2rt_perturber = perturber

    # Credit-assign the previous push for these env_ids using their CURRENT tracking error --
    # enough simulated time has passed (this event's own interval, 1-3s) for a disruptive push's
    # effect to show up in motion_anchor_pos error.
    command = env.command_manager.get_term("motion")
    error_now = command.metrics["error_anchor_pos"][env_ids]
    perturber.finish_pending_and_update(env_ids, error_now)

    # Observe this env's current base state, then sample+apply a new push.
    root_lin_vel_b = asset.data.root_lin_vel_b[env_ids]
    root_ang_vel_b = asset.data.root_ang_vel_b[env_ids]
    height = asset.data.root_pos_w[env_ids, 2:3]
    state = torch.cat([height, root_lin_vel_b, root_ang_vel_b], dim=-1)

    action = perturber.act(env_ids, state)  # (-1, 1) per axis
    ranges = torch.tensor([velocity_range.get(ax, (0.0, 0.0)) for ax in AXES], device=env.device)
    half_width = (ranges[:, 1] - ranges[:, 0]) / 2.0
    center = (ranges[:, 1] + ranges[:, 0]) / 2.0
    scaled = center + action * half_width  # map (-1,1) -> [min,max] per axis, matching the
    # static version's range exactly so N4 is a fair comparison (same max possible push) that
    # merely *learns* where in that range to land instead of sampling it uniformly.

    vel_w = asset.data.root_vel_w[env_ids].clone()
    vel_w[:, :3] += scaled[:, :3]
    vel_w[:, 3:] += scaled[:, 3:]
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
