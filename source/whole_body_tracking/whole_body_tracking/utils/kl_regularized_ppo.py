"""KL-regularized PPO for fine-tuning stabilization (condition B).

Motivation (see docs/강화학습_방법_및_필독논문.md 4-5절 / 연구확장및졸업논문계획서.md 3-2절,
개선 후보 ⑤): RL's Razor (Shenfeld, Pari & Agrawal, MIT Improbable AI Lab, 2025,
arXiv:2509.04259) shows that on-policy RL implicitly prefers, among the many solutions to
a new task, the one closest in KL divergence to the policy it started from -- which is
argued to be why RL forgets less than supervised fine-tuning. This module makes that bias
*explicit and tunable*: it adds `kl_coef * KL(pi_new(.|s) || pi_reference(.|s))` to the PPO
loss, where `pi_reference` is a frozen copy of the checkpoint condition B is warm-started
from (i.e. condition A's pretrained policy). The intent is to reduce condition B's
seed-to-seed variance and catastrophic forgetting on AMASS, seen in Stage 2 (seed 1 broke
the RQ2 "B beats C" trend that held in seeds 0 and 2 -- docs/진행상황_연구노트.md 0절).

Not a standalone algorithm: `attach_kl_regularization()` upgrades an already-constructed
`rsl_rl.algorithms.ppo.PPO` instance in place, so it plugs into the existing
`MotionOnPolicyRunner` without touching rsl_rl itself.
"""

import copy

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class KLRegularizedPPO(PPO):
    """`rsl_rl` PPO with an added `kl_coef * KL(pi_new || pi_reference)` penalty.

    Never constructed directly -- `attach_kl_regularization()` reassigns an existing
    `PPO` instance's `__class__` to this subclass, which is enough since the only new
    state (`reference_policy`, `kl_coef`) is set by `set_reference_policy()` right after.
    """

    def set_reference_policy(self, reference_policy: nn.Module, kl_coef: float):
        self.reference_policy = reference_policy
        self.kl_coef = kl_coef

    def update(self):  # noqa: C901
        # Forked from rsl_rl.algorithms.ppo.PPO.update() (rsl_rl 2.x) to add the
        # reference-KL penalty below (marked "RL's Razor"). Keep in sync with upstream
        # if rsl_rl is upgraded -- everything except that block is an unmodified copy.
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_kl_reference = 0
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

            # --- RL's Razor: explicit KL regularization toward the frozen reference policy ---
            # Reverse-KL(new || reference), same direction as the standard RLHF KL penalty
            # (`reward -= beta * KL(pi_theta || pi_ref)`): pulls the fine-tuned policy back
            # toward the pretrained one, making the "stay close in KL" bias RL's Razor
            # identifies as implicit in on-policy RL into an explicit, tunable term.
            with torch.no_grad():
                self.reference_policy.act(obs_batch[:original_batch_size])
                ref_mu = self.reference_policy.action_mean
                ref_sigma = self.reference_policy.action_std
            kl_reference = torch.sum(
                torch.log(ref_sigma / sigma_batch + 1.0e-5)
                + (torch.square(sigma_batch) + torch.square(mu_batch - ref_mu)) / (2.0 * torch.square(ref_sigma))
                - 0.5,
                axis=-1,
            )
            kl_reference_mean = kl_reference.mean()
            loss = loss + self.kl_coef * kl_reference_mean

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
            mean_kl_reference += kl_reference_mean.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_kl_reference /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_reference": mean_kl_reference,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


def attach_kl_regularization(runner, reference_checkpoint: str, kl_coef: float) -> None:
    """Upgrade `runner.alg` (a plain rsl_rl `PPO`) to `KLRegularizedPPO` in place.

    Call this *after* `runner.load(reference_checkpoint)` has already warm-started the
    trainable policy from the same checkpoint (condition B's normal resume-from-A step),
    so the reference and the trainable policy start out identical and only diverge as
    fine-tuning proceeds -- that divergence is exactly what the KL term penalizes.
    """
    alg = runner.alg
    alg.__class__ = KLRegularizedPPO

    reference_policy = copy.deepcopy(alg.policy)
    checkpoint = torch.load(reference_checkpoint, map_location=alg.device, weights_only=False)
    reference_policy.load_state_dict(checkpoint["model_state_dict"])
    reference_policy.to(alg.device)
    reference_policy.eval()
    for param in reference_policy.parameters():
        param.requires_grad_(False)

    alg.set_reference_policy(reference_policy, kl_coef)
    print(f"[INFO] KL regularization enabled (RL's Razor): reference={reference_checkpoint}, kl_coef={kl_coef}")
