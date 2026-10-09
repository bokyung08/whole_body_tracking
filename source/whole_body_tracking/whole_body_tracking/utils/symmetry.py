"""G1 추종 과제를 위한 좌우 대칭 보조손실 (N2 후보).

배경 (문헌 조사, Phase 10 재설계; g1-fullscale-tracking의
docs/진행상황_연구노트.md 2026-10-08 "새 개선 기법 후보 탐색" 참고): Mittal 외,
"Symmetry Considerations for Learning Task Symmetric Robot Policies"(2024, arXiv:2403.04359)에
따르면, 대칭 신경망 구조를 직접 설계하는 대신 pi(mirror(obs)) ~= mirror(pi(obs))를 보조 MSE
손실로 강제하는 것만으로도 이족보행 로봇의 좌우 대칭성을 활용하는 이득 대부분을 거의 구현
비용 없이 얻을 수 있다. IsaacLab의 rsl_rl 연동(isaaclab_rl.rsl_rl.symmetry_cfg.RslRlSymmetryCfg)이
학습 루프 쪽(rsl_rl의 PPO.update() 안 "Symmetry loss" 블록)은 이미 구현해 두었으므로, 이 과제에서
직접 만들어야 하는 건 G1 전용 거울변환 함수 하나뿐이고, 그게 이 파일이다.

아래 관절/바디 순서는 flat_env_cfg.py나 로봇 USD/MJCF 선언 순서를 "추측"한 게 아니라, 내보낸
정책의 ONNX 메타데이터(`onnx.load(...).metadata_props`)에서 직접 읽어 확인한 값이다. 이 프로젝트는
예전에 정확히 이 가정(선언 순서 = 실행 순서) 때문에 틀린 적이 있다(humanoid-amass-kit의
docs/데이터_분석.md, raw-vs-processed 절 참고) — Isaac Sim의 articulation 빌더는 실행 시점에
자체 순회 방식으로 관절 순서를 다시 매기기 때문에, 소스 코드의 선언 순서와 실제 런타임 순서가
다를 수 있다. ONNX의 `joint_names`/`body_names` 메타데이터는 내보내기 시점에
`env.scene["robot"].data.joint_names` / `.cfg.body_names`에서 그대로 가져온 값(utils/exporter.py)이라
실제 런타임 순서를 정확히 반영하며, 따라서 이 값을 근거로 삼는다. 로봇 USD나 `MotionCommandCfg`가
바뀌면 최신 ONNX를 다시 내보내 이 표를 재검증해야 한다.

부호 규약: 아래의 모든 항목은 이미 로봇 자신의 base/anchor 좌표계 기준으로 표현되어 있다
(IsaacLab의 `base_lin_vel`/`base_ang_vel`은 `root_*_b`이고, `motion_anchor_*_b`와
`robot_body_*_b`도 `subtract_frame_transforms`를 거친 anchor 좌표계 값이다). 따라서 거울변환은
로봇의 현재 월드 방향(heading)과 무관하게 항상 "로컬 좌우(Y) 축과 그에 얽힌 항들의 부호를
뒤집는" 고정된 변환이 된다.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------------------------
# 관절 표 (총 29개). PERM[i] = 거울변환된 벡터의 i번째 값을 만들 때 "어느 인덱스에서 가져올지",
# SIGN[i] = 그때 곱할 부호. pitch류/힌지류 관절(hip/shoulder/ankle/wrist의 pitch, knee, elbow)은
# 좌우 거울변환에 대해 대칭이라 "좌우만 바꿔치기, 부호는 그대로" 되고, roll류·yaw류 관절은
# 반대칭이라 "바꿔치기 + 부호 반전"이 된다 — 이는 내보낸 정책의 `default_joint_pos` 메타데이터로
# 실측 확인됨: 예를 들어 좌/우 shoulder_roll의 기본값이 +0.200/-0.200로 부호가 반대인데, 이는
# "좌우로 거울 대칭인 정지자세"에서 기대되는 그대로다. 중앙 관절(waist_*)은 거울 짝이 없고
# (PERM[i] = i), roll류·yaw류라면 부호만 뒤집는다.
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
    +1, +1,  # hip_pitch 좌,우
    -1,  # waist_yaw
    -1, -1,  # hip_roll 좌,우
    -1,  # waist_roll
    -1, -1,  # hip_yaw 좌,우
    +1,  # waist_pitch
    +1, +1,  # knee 좌,우
    +1, +1,  # shoulder_pitch 좌,우
    +1, +1,  # ankle_pitch 좌,우
    -1, -1,  # shoulder_roll 좌,우
    -1, -1,  # ankle_roll 좌,우
    -1, -1,  # shoulder_yaw 좌,우
    +1, +1,  # elbow 좌,우
    -1, -1,  # wrist_roll 좌,우
    +1, +1,  # wrist_pitch 좌,우
    -1, -1,  # wrist_yaw 좌,우
]
assert len(JOINT_NAMES) == len(JOINT_PERM) == len(JOINT_SIGN) == 29
# involution(대합) 검증: 같은 관절을 두 번 거울변환하면 원래 값으로 돌아와야 하고, 그때 부호를
# 두 번 곱한 값은 반드시 +1이어야 한다.
assert all(JOINT_PERM[JOINT_PERM[i]] == i for i in range(29)), "JOINT_PERM이 자기 자신의 역함수가 아님"
assert all(JOINT_SIGN[i] * JOINT_SIGN[JOINT_PERM[i]] == 1 for i in range(29)), "JOINT_SIGN이 JOINT_PERM과 모순됨"


def _check_joint_names(names: list[str]) -> None:
    """실제 런타임 관절 순서가 위 표(JOINT_NAMES)와 일치하는지 확인한다. 로봇/환경 설정이 바뀌어
    런타임 순서가 달라졌는데 표를 갱신하지 않으면, 거울변환이 조용히 엉뚱한 관절끼리 바꿔치기해
    학습 신호를 오염시킬 수 있다 — 그런 사고를 막기 위한 가드."""
    if list(names) != JOINT_NAMES:
        raise ValueError(
            "symmetry.py의 JOINT_NAMES 표가 실제 런타임 관절 순서와 다릅니다 "
            f"(받은 값: {names!r}). 대칭 손실을 켜기 전에, 최신 ONNX로 내보낸 정책의 "
            "`joint_names` 메타데이터를 기준으로 이 표를 다시 만드세요."
        )


# 바디 표 (추적 대상 14개 바디, MotionCommandCfg.body_names / ONNX `body_names` 메타데이터
# 순서). 정책(policy) 관측에는 바디별 위치/자세가 포함되지 않고 anchor 하나만 들어가므로,
# 이 표는 크리틱(critic)의 robot_body_pos_b/robot_body_ori_b 항에만 쓰인다.
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
assert all(BODY_PERM[BODY_PERM[i]] == i for i in range(14)), "BODY_PERM이 자기 자신의 역함수가 아님"

# 정책/크리틱 관측 벡터의 평탄화된 레이아웃(항목 이름 -> 차원 수), tracking_env_cfg.py의
# ObservationsCfg.PolicyCfg / CriticCfg에 선언된 순서 그대로다 — IsaacLab의 ObsGroup은 항목을
# "선언된 순서대로" 이어붙인다. `command`는 MotionCommand.command, 즉
# cat([참조 joint_pos(29), 참조 joint_vel(29)])다(commands.py 참고).
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
    """(..., 29) 모양의 관절 공간 블록(위치·속도·행동 중 하나)을 거울변환한다."""
    perm = torch.as_tensor(JOINT_PERM, device=x.device, dtype=torch.long)
    sign = torch.as_tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)
    return x.index_select(-1, perm) * sign


def _mirror_vec3(x: torch.Tensor) -> torch.Tensor:
    """로컬 좌표계 기준 3차원 벡터(위치 또는 선속도)를 거울변환한다: 좌우(Y, 인덱스 1) 축만 부호 반전."""
    sign = torch.tensor([1.0, -1.0, 1.0], device=x.device, dtype=x.dtype)
    return x * sign


def _mirror_ang_vel3(x: torch.Tensor) -> torch.Tensor:
    """로컬 좌표계 기준 각속도를 거울변환한다: roll·yaw 축(x, z)은 부호가 뒤집히고,
    pitch 축(y)은 그대로다(선형량과 반대 패턴 — 각속도는 축성 벡터라 반사 변환에서
    부호 규칙이 위치/선속도와 다르게 적용된다)."""
    sign = torch.tensor([-1.0, 1.0, -1.0], device=x.device, dtype=x.dtype)
    return x * sign


def _mirror_rot6(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) 모양의 6D 회전 표현(3x3 회전행렬의 앞 두 열을, row-major로 평탄화한
    [R00,R01,R10,R11,R20,R21] — mdp/observations.py의 `mat[..., :2].reshape(...)` 참고)을
    거울변환한다. 반사행렬 F=diag(1,-1,1)에 대해 R' = F @ R @ F가 성립하므로, (행, 열) 중
    정확히 하나만 Y축(인덱스 1)인 원소의 부호가 뒤집힌다: R01, R10, R21
    (평탄화된 인덱스로는 1, 2, 4번)."""
    sign = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], device=x.device, dtype=x.dtype)
    return x * sign


def _mirror_command(x: torch.Tensor) -> torch.Tensor:
    """MotionCommand.command = cat([참조 joint_pos(29), 참조 joint_vel(29)])를 거울변환한다."""
    ref_pos, ref_vel = x[..., :29], x[..., 29:]
    return torch.cat([_mirror_joint_block(ref_pos), _mirror_joint_block(ref_vel)], dim=-1)


def _mirror_body_block(x: torch.Tensor, per_body_width: int, mirror_fn) -> torch.Tensor:
    """(..., 14 * per_body_width) 모양으로 14개 바디가 이어붙은 블록을 거울변환한다:
    먼저 좌우 바디 쌍을 맞바꾸고(BODY_PERM), 그 다음 각 바디 자신의 per_body_width 크기
    슬라이스에 `mirror_fn`을 적용한다."""
    *lead, _ = x.shape
    blocks = x.view(*lead, 14, per_body_width)
    perm = torch.as_tensor(BODY_PERM, device=x.device, dtype=torch.long)
    blocks = blocks.index_select(-2, perm)
    blocks = mirror_fn(blocks)
    return blocks.reshape(*lead, 14 * per_body_width)


def mirror_observation(obs: torch.Tensor, obs_type: str) -> torch.Tensor:
    """(..., POLICY_WIDTH) 또는 (..., CRITIC_WIDTH) 모양의 평탄화된 관측 벡터를 거울변환한다."""
    terms = POLICY_TERMS if obs_type == "policy" else CRITIC_TERMS
    width = POLICY_WIDTH if obs_type == "policy" else CRITIC_WIDTH
    if obs.shape[-1] != width:
        raise ValueError(f"obs_type={obs_type!r}는 차원 수 {width}를 기대하는데 {obs.shape[-1]}이 들어왔습니다 — 관측 레이아웃이 바뀌었나요?")

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
            raise AssertionError(f"처리되지 않은 관측 항목입니다: {name!r}")
    return torch.cat(out, dim=-1)


def mirror_action(actions: torch.Tensor) -> torch.Tensor:
    """(..., 29) 모양의 행동 벡터(관절 목표 위치, joint_pos와 같은 레이아웃)를 거울변환한다."""
    return _mirror_joint_block(actions)


def g1_tracking_mirror_augmentation(env, obs, actions, obs_type: str = "policy"):
    """`isaaclab_rl.rsl_rl.symmetry_cfg.RslRlSymmetryCfg`가 요구하는 `data_augmentation_func`.

    rsl_rl의 PPO.update()가 `use_mirror_loss`일 때 호출하는 방식(rsl_rl/algorithms/ppo.py의
    "Symmetry loss" 블록 참고)을 그대로 따른다: `obs`만 넘기면 `cat([obs, mirror(obs)])`를
    첫 번째 반환값으로 주고(정책망이 원본과 거울상을 한 번에 평가할 수 있도록), `actions`만
    넘기면 `cat([actions, mirror(actions)])`를 두 번째 반환값으로 준다 — 두 경우 모두
    "첫 번째 증강이 곧 원본"이라는 rsl_rl의 규약을 따른다.
    """
    aug_obs = None
    aug_actions = None
    if obs is not None:
        aug_obs = torch.cat([obs, mirror_observation(obs, obs_type)], dim=0)
    if actions is not None:
        aug_actions = torch.cat([actions, mirror_action(actions)], dim=0)
    return aug_obs, aug_actions
