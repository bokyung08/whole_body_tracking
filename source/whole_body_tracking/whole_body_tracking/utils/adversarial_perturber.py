"""N4(SA2RT 스타일 선택적 적대 교란)의 순수 torch 핵심 로직 — isaaclab을 import하지 않으므로
Isaac Sim 없이도 단독으로 단위 테스트가 가능하다(g1-fullscale-tracking의
scripts/_test_adversarial_perturber.py 참고). 이걸 실제로 쓰는 환경-이벤트 래퍼
(tasks/tracking/mdp/adversarial_push.py)는 isaaclab.assets/managers가 필요해서 Isaac Sim
없이는 단독 import가 안 된다(commands.py가 이미 갖고 있는 것과 같은 제약).

전체 설계 근거(왜 완전한 교대 bi-level RL이 아니라 1-step REINFORCE 기반 contextual bandit을
택했는지)는 adversarial_push.py의 모듈 docstring을 참고.
"""

import torch
import torch.nn as nn


class AdversarialPerturber(nn.Module):
    """로봇 자신의 base 상태(높이, 선속도, 각속도 — 7차원)를 보고 6개 push-속도 축에 대한 값을
    내놓는 아주 작은 대각 가우시안 정책. 1-step REINFORCE로 온라인 학습한다: 각 push의 보상은
    "그 환경(env)에 대해 다음번에 이 함수가 호출될 때" 관측되는 추종 오차(= push 효과가 나타날
    만큼 시간이 지난 뒤의 오차)에서, push 크기에 대한 선택성 페널티를 뺀 값이다. 이 페널티
    덕분에 "효과도 없을 push인데 무작정 세게 미는" 쪽으로 수렴하지 않고, 효과 없는 상황에서는
    살살 미는 법을 학습한다."""

    def __init__(self, num_envs: int, device: torch.device, selectivity_coef: float = 0.01, lr: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 12)).to(
            device
        )
        self.log_std = nn.Parameter(torch.zeros(6, device=device) - 1.0)  # 초기 std = exp(-1) ≈ 0.37
        self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=lr)
        self.selectivity_coef = selectivity_coef
        self.device = device

        # 아래 세 버퍼는 의도적으로 autograd 추적을 끈(detach된) 단순 기록용 버퍼다. log_prob은
        # act() 호출 시점에 캐싱해두는 대신 finish_pending_and_update()에서 매번 새로 계산한다
        # (grad를 켠 채로). 이유: grad가 걸린 값을 같은 영속(persistent) 텐서에 인덱싱으로
        # 반복해서 덮어쓰면(`buf[ids] = value`), 그 버퍼의 과거 모든 쓰기 이력이 하나의 점점
        # 자라나는 계산 그래프로 체인되어 버린다 — 그러면 나중 배치를 backward()할 때, 이전
        # backward()가 이미 해제해버린 노드를 다시 거슬러 올라가려다 "Trying to backward through
        # the graph a second time" 에러가 난다. detach된 (state, action)에서 매번 새로 계산하면
        # 이 문제가 애초에 생기지 않는다 — 매 update마다 그래프가 완전히 새것이고 독립적이다.
        self.has_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pending_state = torch.zeros(num_envs, 7, device=device)
        self.pending_action = torch.zeros(num_envs, 6, device=device)
        self.reward_baseline = 0.0
        self._buffer_log_probs: list[torch.Tensor] = []
        self._buffer_rewards: list[torch.Tensor] = []
        self._updates_since_step = 0
        self.update_every = 8  # push 이벤트 약 8번마다 한 번씩 옵티마이저 스텝을 밟는다

    def _policy(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.net(state)[:, :6])  # (-1, 1) 범위, 실제 속도 범위로의 변환은 호출하는 쪽(adversarial_push.py)이 담당
        std = torch.exp(self.log_std).clamp(min=1e-3).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def finish_pending_and_update(self, env_ids: torch.Tensor, error_now: torch.Tensor):
        """env_ids 중 이전에 보류(pending) 중이던 push가 있으면 보상을 매겨 REINFORCE 샘플로
        쌓는다(log_prob은 저장해둔 detached state/action에서 매번 새로 계산 — 이유는 __init__
        주석 참고). 같은 env_ids에 대해 act()로 pending_state/pending_action을 덮어쓰기 전에
        반드시 먼저 호출해야 한다."""
        mask = self.has_pending[env_ids]
        if not torch.any(mask):
            return
        settled_ids = env_ids[mask]
        dist = self._policy(self.pending_state[settled_ids])
        log_prob = dist.log_prob(self.pending_action[settled_ids]).sum(dim=-1)
        reward = error_now[mask] - self.selectivity_coef * self.pending_action[settled_ids].pow(2).sum(dim=-1)
        self._buffer_log_probs.append(log_prob)
        self._buffer_rewards.append(reward.detach())

        self._updates_since_step += 1
        if self._updates_since_step >= self.update_every and self._buffer_rewards:
            self._apply_update()
            self._updates_since_step = 0

    def _apply_update(self):
        """쌓아둔 REINFORCE 샘플로 실제 경사하강 1스텝을 밟는다. 베이스라인(이동평균)을 빼서
        분산을 줄인 advantage로 정책 경사(policy gradient)를 계산한다."""
        log_probs = torch.cat(self._buffer_log_probs)
        rewards = torch.cat(self._buffer_rewards)
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * rewards.mean().item()
        advantage = rewards - self.reward_baseline
        loss = -(log_probs * advantage).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._buffer_log_probs.clear()
        self._buffer_rewards.clear()

    def act(self, env_ids: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """env_ids에 대해 새 push 행동((-1, 1) 범위)을 샘플링하고, 나중에
        finish_pending_and_update()가 log_prob을 다시 계산할 수 있도록 (detach해서) 보류
        상태로 기억해둔다."""
        with torch.no_grad():
            action = self._policy(state).sample()
        self.pending_state[env_ids] = state.detach()
        self.pending_action[env_ids] = action
        self.has_pending[env_ids] = True
        return action
