"""Left-right mirror symmetry auxiliary loss for the G1 tracking task (candidate N2).

Motivation (literature survey, Phase 10 redesign; see g1-fullscale-tracking's
docs/진행상황_연구노트이.md 2026-10-08 "새 개선 기법 후보 탐색"): Mittal et al., "Symmetry
Considerations for Learning Task Symmetric Robot Policies" (2024, arXiv:2403.04359) show that
enforcing pi(mirror(obs)) ~= mirror(pi(obs)) as an auxiliary MSE loss -- rather than hand-designing
a symmetric network architecture -- captures most of the benefit of exploiting a bipedal robot's
exact left-right symmetry, at near-zero implementation cost. IsaacLab's rsl_rl integration
(isaaclab_rl.rsl_rl.symmetry_cfg.RslRlSymmetryCfg) already implements the training-loop side of
this (rsl_rl's PPO.update(), "Symmetry loss" block) -- the only missing, task-specific piece is
the mirror transform itself, which is what this file provides.

Ground truth for the joint/body order below was read directly from an exported policy's ONNX
metadata (`onnx.load(...).metadata_props`), NOT assumed from flat_env_cfg.py's or the robot's
USD/MJCF declaration order. This project was burned before by exactly that assumption (see
humanoid-amass-kit's docs/데이터_분석.md, raw-vs-processed section): Isaac Sim's articulation
builder reorders joints at runtime via its own traversal, not source-declaration order. The ONNX
`joint_names`/`body_names` metadata is populated from `env.scene["robot"].data.joint_names` /
`.cfg.body_names` at export time (utils/exporter.py), i.e. the actual runtime order, so it is
authoritative. Re-verify against a fresh ONNX export if the robot USD or `MotionCommandCfg`
changes.

Sign convention: all terms below are already expressed in the robot's own base/anchor frame
(IsaacLab's `base_lin_vel`/`base_ang_vel` are `root_*_b`; `motion_anchor_*_b` and `robot_body_*_b`
are anchor-frame via `subtract_frame_transforms`), so the mirror is a fixed local transform
(negate the local lateral/Y axis and its cross-terms) that does not depend on the robot's current
world heading.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------------------------
# Joint table (29 joints). PERM[i] = index to pull FROM when building the mirrored vector's
# entry i; SIGN[i] = the sign to apply. Pitch/hinge joints (hip/shoulder/ankle/wrist pitch, knee,
# elbow) are symmetric under a left-right mirror (swap partner, keep sign); roll and yaw joints
# are anti-symmetric (swap partner, flip sign) -- confirmed against the exported policy's
# `default_joint_pos` metadata, where e.g. left/right shoulder_roll defaults are +0.200/-0.200
# (opposite signs for a physically mirror-symmetric resting pose), matching this rule. Central
# joints (waist_*) have no partner (PERM[i] = i) and flip sign iff they are roll/yaw.
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
#         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
JOINT_PERM = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27]
JOINT_SIGN = [
    +1, +1,  # hip_pitch L,R
    -1,  # waist_yaw
    -1, -1,  # hip_roll L,R
    -1,  # waist_roll
    -1, -1,  # hip_yaw L,R
    +1,  # waist_pitch
    +1, +1,  # knee L,R
    +1, +1,  # shoulder_pitch L,R
    +1, +1,  # ankle_pitch L,R
    -1, -1,  # shoulder_roll L,R
    -1, -1,  # ankle_roll L,R
    -1, -1,  # shoulder_yaw L,R
    +1, +1,  # elbow L,R
    -1, -1,  # wrist_roll L,R
    +1, +1,  # wrist_pitch L,R
    -1, -1,  # wrist_yaw L,R
]
assert len(JOINT_NAMES) == len(JOINT_PERM) == len(JOINT_SIGN) == 29
# involution check: mirroring twice must return the original joint, with signs multiplying to +1
assert all(JOINT_PERM[JOINT_PERM[i]] == i for i in range(29)), "JOINT_PERM is not its own inverse"
assert all(JOINT_SIGN[i] * JOINT_SIGN[JOINT_PERM[i]] == 1 for i in range(29)), "JOINT_SIGN inconsistent with JOINT_PERM"


def _check_joint_names(names: list[str]) -> None:
    if list(names) != JOINT_NAMES:
        raise ValueError(
            "symmetry.py's JOINT_NAMES table does not match the runtime joint order "
            f"(got {names!r}). Re-derive the table from a fresh ONNX export's "
            "`joint_names` metadata before enabling the symmetry loss."
        )


# Body table (14 tracked bodies, order per MotionCommandCfg.body_names / ONNX `body_names`
# metadata). Used only by the critic's robot_body_pos_b/robot_body_ori_b terms (policy obs does
# not include per-body position/orientation, only the single anchor).
BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
BODY_PERM = [0, 4, 5, 6, 1, 2, 3, 7, 11, 12, 13, 8, 9, 10]
assert len(BODY_NAMES) == len(BODY_PERM) == 14
assert all(BODY_PERM[BODY_PERM[i]] == i for i in range(14)), "BODY_PERM is not its own inverse"

# Flat policy/critic observation layout (term name -> width), in the exact order the terms are
# declared in tracking_env_cfg.py's ObservationsCfg.PolicyCfg / CriticCfg -- IsaacLab's ObsGroup
# concatenates terms in declaration order. `command` is MotionCommand.command = cat([ref joint_pos
# (29), ref joint_vel (29)]) (commands.py).
POLICY_TERMS = [
    ("command", 58), ("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6),
    ("base_lin_vel", 3), ("base_ang_vel", 3),
    ("joint_pos", 29), ("joint_vel", 29), ("actions", 29),
]
CRITIC_TERMS = [
    ("command", 58), ("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6),
    ("body_pos", 3 * 14), ("body_ori", 6 * 14),
    ("base_lin_vel", 3), ("base_ang_vel", 3),
    ("joint_pos", 29), ("joint_vel", 29), ("actions", 29),
]
POLICY_WIDTH = sum(w for _, w in POLICY_TERMS)  # 160
CRITIC_WIDTH = sum(w for _, w in CRITIC_TERMS)  # 286


def _mirror_joint_block(x: torch.Tensor) -> torch.Tensor:
    """Mirror a (..., 29) joint-space block (pos, vel, or action)."""
    perm = torch.as_tensor(JOINT_PERM, device=x.device, dtype=torch.long)
    sign = torch.as_tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)
    return x.index_select(-1, perm) * sign


def _mirror_vec3(x: torch.Tensor) -> torch.Tensor:
    """Mirror a local-frame 3-vector (position or linear velocity): negate the Y (index 1) axis."""
    sign = torch.tensor([1.0, -1.0, 1.0], device=x.device, dtype=x.dtype)
    return x * sign


def _mirror_ang_vel3(x: torch.Tensor) -> torch.Tensor:
    """Mirror a local-frame angular velocity: roll/yaw rates (x, z) flip, pitch rate (y) doesn't."""
    sign = torch.tensor([-1.0, 1.0, -1.0], device=x.device, dtype=x.dtype)
    return x * sign


def _mirror_rot6(x: torch.Tensor) -> torch.Tensor:
    """Mirror a (..., 6) 6D rotation repr (first two columns of a 3x3 matrix, row-major flat as
    [R00,R01,R10,R11,R20,R21] -- see mdp/observations.py's `mat[..., :2].reshape(...)`), under a
    reflection F=diag(1,-1,1): R' = F @ R @ F, i.e. negate entries where exactly one of (row, col)
    is the Y axis (row or col index 1): R01, R10, R21 (flat indices 1, 2, 4).
    """
    sign = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], device=x.device, dtype=x.dtype)
    return x * sign


def _mirror_command(x: torch.Tensor) -> torch.Tensor:
    """Mirror MotionCommand.command = cat([ref_joint_pos(29), ref_joint_vel(29)])."""
    ref_pos, ref_vel = x[..., :29], x[..., 29:]
    return torch.cat([_mirror_joint_block(ref_pos), _mirror_joint_block(ref_vel)], dim=-1)


def _mirror_body_block(x: torch.Tensor, per_body_width: int, mirror_fn) -> torch.Tensor:
    """Mirror a (..., 14 * per_body_width) stacked per-body block: swap body pairs (BODY_PERM)
    then apply `mirror_fn` to each body's own per_body_width-sized slice."""
    *lead, _ = x.shape
    blocks = x.view(*lead, 14, per_body_width)
    perm = torch.as_tensor(BODY_PERM, device=x.device, dtype=torch.long)
    blocks = blocks.index_select(-2, perm)
    blocks = mirror_fn(blocks)
    return blocks.reshape(*lead, 14 * per_body_width)


def mirror_observation(obs: torch.Tensor, obs_type: str) -> torch.Tensor:
    """Mirror a flat (..., POLICY_WIDTH) or (..., CRITIC_WIDTH) observation vector."""
    terms = POLICY_TERMS if obs_type == "policy" else CRITIC_TERMS
    width = POLICY_WIDTH if obs_type == "policy" else CRITIC_WIDTH
    if obs.shape[-1] != width:
        raise ValueError(f"obs_type={obs_type!r} expects width {width}, got {obs.shape[-1]} -- did the observation layout change?")

    out = []
    offset = 0
    for name, w in terms:
        block = obs[..., offset : offset + w]
        offset += w
        if name == "command":
            out.append(_mirror_command(block))
        elif name == "motion_anchor_pos_b":
            out.append(_mirror_vec3(block))
        elif name == "motion_anchor_ori_b":
            out.append(_mirror_rot6(block))
        elif name == "base_lin_vel":
            out.append(_mirror_vec3(block))
        elif name == "base_ang_vel":
            out.append(_mirror_ang_vel3(block))
        elif name in ("joint_pos", "joint_vel", "actions"):
            out.append(_mirror_joint_block(block))
        elif name == "body_pos":
            out.append(_mirror_body_block(block, 3, _mirror_vec3))
        elif name == "body_ori":
            out.append(_mirror_body_block(block, 6, _mirror_rot6))
        else:
            raise AssertionError(f"unhandled observation term {name!r}")
    return torch.cat(out, dim=-1)


def mirror_action(actions: torch.Tensor) -> torch.Tensor:
    """Mirror a (..., 29) action vector (joint position targets, same layout as joint_pos)."""
    return _mirror_joint_block(actions)


def g1_tracking_mirror_augmentation(env, obs, actions, obs_type: str = "policy"):
    """`data_augmentation_func` for `isaaclab_rl.rsl_rl.symmetry_cfg.RslRlSymmetryCfg`.

    Matches the calling convention rsl_rl's PPO.update() uses for `use_mirror_loss` (see
    rsl_rl/algorithms/ppo.py's "Symmetry loss" block): called once with only `obs` set (returns
    `cat([obs, mirror(obs)])` as the first element, so the actor can be queried on both at once),
    and once with only `actions` set (returns `cat([actions, mirror(actions)])` as the second
    element) -- "the first augmentation is the original one" in both cases.
    """
    aug_obs = None
    aug_actions = None
    if obs is not None:
        aug_obs = torch.cat([obs, mirror_observation(obs, obs_type)], dim=0)
    if actions is not None:
        aug_actions = torch.cat([actions, mirror_action(actions)], dim=0)
    return aug_obs, aug_actions
