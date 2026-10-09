"""N4 후보(SA2RT 스타일, arXiv:2507.08303): 이 과제의 고정-무작위 push 이벤트
(`events.py`의 `push_by_setting_velocity` / `tracking_env_cfg.py`의 `push_robot` 이벤트,
1~3초 간격, 고정된 `VELOCITY_RANGE`)를 대체하는, 학습되는 상태조건부 *선택적* 적대 교란 정책.
고정 도메인 랜덤화는 그 push가 실제로 로봇을 불안정하게 만드는지와 무관하게 모든 환경(env)을
똑같은 확률분포로 민다. 여기서는 작은 정책 하나가 "어디를·얼마나 세게" 밀어야 실제로 추종을
방해하는지를 학습하고, 세게 밀어도 메인 정책에 가르칠 게 없는 상태에서는 살살(또는 거의 안)
미는 법도 함께 학습한다 — 그래서 "선택적(selective)"이다.

범위 참고 (계획서의 N4는 이 부분 하나를 가리키며, APEX류의 완전한 bi-level RL이 아니다): 이건
단순화된, 환경-이벤트 수준의 구현이다 — 진짜로 두 정책이 교대로 학습하는 루프를 만들려면
`MotionOnPolicyRunner.learn()` 자체를 포크하고 별도의 rollout buffer도 필요해지는, 훨씬 크고
위험한 변경이 되기 때문에 일부러 피했다. 그래도 이 후보가 실제로 주장하는 핵심(고정된 분포가
아니라, 자기 자신의 보상 신호로 적응하는 *학습된* 교란 정책)은 간단한 contextual-bandit /
1-step 정책경사(REINFORCE) 업데이트로 충분히 만족시킨다 — 메인 PPO 정책의 학습을 막지 않으면서
그와 번갈아(온라인으로) 학습된다. 교란 정책 자체와 그 REINFORCE 업데이트는
whole_body_tracking/utils/adversarial_perturber.py(순수 torch, isaaclab 의존성 없음, 단독
단위테스트 가능)에 있고, 이 파일은 그걸 감싸는 얇은 환경-이벤트 래퍼일 뿐이다.

어떤 env에 대해 이 이벤트가 호출될 때마다(IsaacLab의 interval 모드 EventManager, 기존
고정-push와 같은 1~3초 간격) 두 가지 일을 한다:
  (1) 그 env에 대해 이전에 보류 중이던 push가 있으면 *credit assignment를 마무리*한다: 그
      과거 행동의 보상은 이 env의 *현재* motion_anchor_pos 추종 오차다(= push 효과가 드러날
      만큼 충분히 시간이 지난 뒤 "추종이 얼마나 망가졌는지").
  (2) 업데이트 이후의 최신 교란 정책으로 *새* push를 샘플링해서 적용하고, 다음 호출의 credit
      assignment를 위해 "보류 중"으로 기억해둔다.

`--adversarial_push`를 넘기지 않으면 전혀 쓰이지 않는다(train.py가
env_cfg.events.push_robot.func를 이 모듈의 `adversarial_push`로 바꿔치기하는데, 대체 대상인
고정 `push_by_setting_velocity`와 호출 시그니처(`params={"velocity_range": ...}`)가 완전히
같아서 env_cfg 구조를 바꿀 필요가 없다).
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
    """isaaclab.envs.mdp.push_by_setting_velocity와 호출 시그니처가 완전히 같은 대체품이라
    `env_cfg.events.push_robot.func =` 한 줄로 바로 바꿔 끼울 수 있다 — 다만 push를 균등 난수로
    뽑는 대신, 작고 학습되는 선택적 적대 정책이 고른다. 자세한 설계는 모듈 docstring 참고."""
    asset: "RigidObject | Articulation" = env.scene[asset_cfg.name]

    perturber: AdversarialPerturber | None = getattr(env, "_sa2rt_perturber", None)
    if perturber is None:
        perturber = AdversarialPerturber(env.num_envs, env.device)
        env._sa2rt_perturber = perturber

    # 이 env_ids의 "이전" push에 현재 추종 오차로 보상을 매긴다 — 이 이벤트 자신의 호출 간격
    # (1~3초)만큼 시뮬레이션 시간이 지났으니, push가 실제로 방해가 됐다면 그 효과가
    # motion_anchor_pos 오차에 이미 드러났을 시간이다.
    command = env.command_manager.get_term("motion")
    error_now = command.metrics["error_anchor_pos"][env_ids]
    perturber.finish_pending_and_update(env_ids, error_now)

    # 이 env의 현재 base 상태를 관측한 뒤, 새 push를 샘플링해서 적용한다.
    root_lin_vel_b = asset.data.root_lin_vel_b[env_ids]
    root_ang_vel_b = asset.data.root_ang_vel_b[env_ids]
    height = asset.data.root_pos_w[env_ids, 2:3]
    state = torch.cat([height, root_lin_vel_b, root_ang_vel_b], dim=-1)

    action = perturber.act(env_ids, state)  # 축마다 (-1, 1) 범위
    ranges = torch.tensor([velocity_range.get(ax, (0.0, 0.0)) for ax in AXES], device=env.device)
    half_width = (ranges[:, 1] - ranges[:, 0]) / 2.0
    center = (ranges[:, 1] + ranges[:, 0]) / 2.0
    # (-1,1) 범위를 축마다 [min,max]로 매핑한다 — 기존 고정-push와 "최대로 밀 수 있는 세기"
    # 자체는 똑같이 맞춰서, N4가 비교하는 것은 오직 "그 범위 안 어디로 착지시킬지를 학습해서
    # 고르느냐"일 뿐, 범위 자체를 키워서 생긴 이득이 섞이지 않도록(공정한 비교) 했다.
    scaled = center + action * half_width

    vel_w = asset.data.root_vel_w[env_ids].clone()
    vel_w[:, :3] += scaled[:, :3]
    vel_w[:, 3:] += scaled[:, 3:]
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
