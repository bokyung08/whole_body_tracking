"""감쇠하는 시연-행동 사전분포(action prior)를 더한 PPO (N3 후보, APEX 스타일).

배경 (문헌 조사, Phase 10 재설계; g1-fullscale-tracking의 docs/진행상황_연구노트.md 2026-10-08
신규 후보 목록 참고): APEX(arXiv:2505.10022, 2025)는 학습 초반에 시연(demonstration) 조건부
행동 사전분포를 더해준다 — 여기서는 이 과제가 매 스텝 이미 가지고 있는 참조 동작의 목표 관절
자세를 그대로 "시연"으로 쓰므로 별도의 시연 데이터가 필요 없다 — 그리고 그 영향력을 0까지
서서히 줄여서, 학습이 끝날 때는 사전분포 편향이 전혀 없는 순수 RL이 되게 한다. 의도는 학습
초반(정책 자신의 가치 추정이 아직 쓸모없을 때)에 더 빠르고 안정적인 학습 신호를 주되, 수렴한
정책에는 편향을 남기지 않는 것이다. (범위 참고: 계획서의 N3는 이 "감쇠하는 행동 사전분포"
부분만을 가리키며, APEX 원 논문의 별도 2번째 과업/스타일 critic head는 포함하지 않는다 — 그건
rsl_rl의 rollout storage 구조 자체를 바꿔야 하는 훨씬 크고 위험한 변경이라 이번 스크리닝
후보에서는 제외했다. 자세한 내용은 진행상황_연구노트.md의 2026-10-09 "N3 범위" 항목 참고.)

독립된 알고리즘이 아니다: `attach_action_prior()`가 이미 만들어진 `rsl_rl.algorithms.ppo.PPO`
인스턴스를 제자리에서(in place) 이 서브클래스로 업그레이드한다 — kl_regularized_ppo.py와
같은 패턴이다.
"""

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class ActionPriorPPO(PPO):
    """`rsl_rl`의 PPO에 선형으로 감쇠하는 `coef(t) * MSE(mu, prior_action)` 항을 더한 버전.

    `prior_action`은 `JointPositionAction`(use_default_offset)이 참조 동작의 목표 관절 자세를
    정확히 재현하게 만드는 raw action이다: `processed_action = raw_action * scale +
    default_joint_pos`라는 정방향 공식을 역산하면 `prior_action = (ref_joint_pos -
    default_joint_pos) / scale`이 나온다. 참조 관절 자세는 정책 관측 배치의 `command` 항목
    앞 29개 값에서 바로 읽어오므로(MotionCommand.command / tracking_env_cfg.py의
    ObservationsCfg 참고 — `command = cat([ref_joint_pos(29), ref_joint_vel(29)])`), 별도의
    데이터 배선이 필요 없다.

    직접 생성하지 않는다 — `attach_action_prior()`가 이미 존재하는 `PPO` 인스턴스의
    `__class__`를 이 서브클래스로 바꿔치기한다.
    """

    def set_action_prior(
        self, default_joint_pos: torch.Tensor, action_scale: torch.Tensor, prior_coef0: float, decay_iterations: int
    ):
        self.default_joint_pos = default_joint_pos
        self.action_scale = action_scale
        self.prior_coef0 = prior_coef0
        self.decay_iterations = max(decay_iterations, 1)
        self._prior_iteration = 0

    def update(self):  # noqa: C901
        # rsl_rl.algorithms.ppo.PPO.update()(rsl_rl 2.x)를 그대로 복사해 포크한 것 —
        # kl_regularized_ppo.py의 KLRegularizedPPO와 같은 기반이다. 아래 "APEX" 블록을 제외한
        # 나머지는 원본과 동일하니, rsl_rl이 업그레이드되면 이 부분도 같이 맞춰줘야 한다.
        prior_coef = self.prior_coef0 * max(0.0, 1.0 - self._prior_iteration / self.decay_iterations)

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_action_prior = 0
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            num_aug = 1
            original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # --- APEX (N3): 참조 자세 쪽으로 당기는, 감쇠하는 시연-행동-사전분포 MSE 항 ---
            # `command`는 첫 번째 관측 항목이고(tracking_env_cfg.py 참고), 그 앞 29개 값이
            # 이번 스텝 참조 동작의 joint_pos다(commands.py의 MotionCommand.command 프로퍼티).
            # JointPositionAction의 `processed = raw*scale + default_joint_pos` 공식을 역산해
            # 그 자세를 정확히 재현하는 raw action을 구하고, 정책의 평균 행동을 그 쪽으로
            # 끌어당긴다 — 끌어당기는 세기(계수)는 `decay_iterations`까지 선형으로 0에
            # 수렴하므로, 수렴 시점에는 순수 RL이 된다.
            if prior_coef > 0:
                ref_joint_pos = obs_batch[:original_batch_size, :29]
                prior_action = (ref_joint_pos - self.default_joint_pos) / self.action_scale
                action_prior_loss = torch.nn.functional.mse_loss(mu_batch, prior_action)
                loss = loss + prior_coef * action_prior_loss
            else:
                action_prior_loss = torch.zeros((), device=mu_batch.device)

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_action_prior += action_prior_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_action_prior /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        self._prior_iteration += 1

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "action_prior": mean_action_prior,
            "action_prior_coef": prior_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


def attach_action_prior(env, runner, prior_coef0: float, decay_iterations: int) -> None:
    """`runner.alg`(일반 rsl_rl `PPO`)를 제자리에서(in place) `ActionPriorPPO`로 업그레이드한다.

    `env`는 (RslRlVecEnvWrapper로 감싸져 있을 수 있는) 학습용 환경이다; `scene`/`action_manager`에
    접근하기 위해 `.unwrapped`를 쓰는데, utils/exporter.py가 ONNX 메타데이터를 내보낼 때와
    같은 패턴이다.
    """
    alg = runner.alg
    alg.__class__ = ActionPriorPPO

    base_env = env.unwrapped
    default_joint_pos = base_env.scene["robot"].data.default_joint_pos_nominal.to(alg.device)
    action_scale = base_env.action_manager.get_term("joint_pos")._scale[0].to(alg.device)

    alg.set_action_prior(default_joint_pos, action_scale, prior_coef0, decay_iterations)
    print(
        f"[INFO] N3 행동 사전분포 활성화(APEX 스타일 감쇠하는 시연 사전분포): "
        f"coef0={prior_coef0}, decay_iterations={decay_iterations}"
    )
