"""Decaying demonstration-action-prior PPO (candidate N3, APEX-style).

Motivation (literature survey, Phase 10 redesign; g1-fullscale-tracking's
docs/진행상황_연구노트이.md 2026-10-08 new-candidate list): APEX (arXiv:2505.10022, 2025) adds a
demonstration-conditioned action prior early in training -- here, the reference motion's own
target joint pose, which this task already has on hand every step (no extra demonstration data
needed) -- and decays its influence to zero, so training ends as pure RL with no prior bias. The
intent is faster/more stable early learning (the prior gives useful gradient signal before the
policy's own value estimates are any good) without biasing the converged policy. (Scope note: the
plan document's N3 is this decaying action-prior piece only, not APEX's separate second
task/style critic head -- that is a materially larger, riskier change to rsl_rl's rollout storage
and was left out of this screening candidate; see the 2026-10-09 "N3 scope" note in
진행상황_연구노트.md.)

Not a standalone algorithm: `attach_action_prior()` upgrades an already-constructed
`rsl_rl.algorithms.ppo.PPO` instance in place, same pattern as kl_regularized_ppo.py.
"""

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class ActionPriorPPO(PPO):
    """`rsl_rl` PPO with an added, linearly-decaying `coef(t) * MSE(mu, prior_action)` term.

    `prior_action` is the raw action that would make `JointPositionAction` (use_default_offset)
    reproduce the reference motion's target joint pose exactly: since
    `processed_action = raw_action * scale + default_joint_pos`, inverting gives
    `prior_action = (ref_joint_pos - default_joint_pos) / scale`. The reference joint pose is read
    directly out of the policy observation batch's `command` term (its first 29 entries -- see
    MotionCommand.command / tracking_env_cfg.py's ObservationsCfg -- `command =
    cat([ref_joint_pos(29), ref_joint_vel(29)])`), so no extra data plumbing is needed.

    Never constructed directly -- `attach_action_prior()` reassigns an existing `PPO` instance's
    `__class__` to this subclass.
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
        # Forked from rsl_rl.algorithms.ppo.PPO.update() (rsl_rl 2.x), same base as
        # kl_regularized_ppo.py's KLRegularizedPPO -- everything except the "APEX" block below is
        # an unmodified copy. Keep in sync with upstream if rsl_rl is upgraded.
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

            # --- APEX (N3): decaying demonstration-action-prior MSE toward the reference pose ---
            # `command` is the first observation term (see tracking_env_cfg.py); its first 29
            # entries are the reference motion's joint_pos for this step (commands.py's
            # MotionCommand.command property). Invert JointPositionAction's
            # `processed = raw*scale + default_joint_pos` to get the raw action that would
            # reproduce it exactly, and pull the policy's mean action toward it with a coefficient
            # that linearly decays to 0 by `decay_iterations` -- pure RL at convergence.
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
    """Upgrade `runner.alg` (a plain rsl_rl `PPO`) to `ActionPriorPPO` in place.

    `env` is the (possibly RslRlVecEnvWrapper-wrapped) training env; `.unwrapped` is used to reach
    `scene`/`action_manager`, same pattern as utils/exporter.py's ONNX metadata export.
    """
    alg = runner.alg
    alg.__class__ = ActionPriorPPO

    base_env = env.unwrapped
    default_joint_pos = base_env.scene["robot"].data.default_joint_pos_nominal.to(alg.device)
    action_scale = base_env.action_manager.get_term("joint_pos")._scale[0].to(alg.device)

    alg.set_action_prior(default_joint_pos, action_scale, prior_coef0, decay_iterations)
    print(
        f"[INFO] N3 action prior enabled (APEX-style decaying demonstration prior): "
        f"coef0={prior_coef0}, decay_iterations={decay_iterations}"
    )
