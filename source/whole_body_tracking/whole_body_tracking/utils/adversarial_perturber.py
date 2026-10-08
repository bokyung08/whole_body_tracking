"""Pure-torch core of N4 (SA2RT-style selective adversarial perturber) -- no isaaclab imports, so
it can be unit-tested standalone (see scripts/_test_adversarial_perturber.py in
g1-fullscale-tracking) without a running Isaac Sim app, unlike the environment-event wrapper that
uses it (tasks/tracking/mdp/adversarial_push.py, which needs isaaclab.assets/managers and so
cannot be isolate-imported -- same constraint commands.py already has).

See adversarial_push.py's module docstring for the full design rationale (why a one-step
REINFORCE contextual bandit rather than a true alternating bi-level RL loop).
"""

import torch
import torch.nn as nn


class AdversarialPerturber(nn.Module):
    """Tiny diagonal-Gaussian policy over 6 push-velocity axes, conditioned on the robot's own
    base state (height, linear vel, angular vel -- 7 dims). Trained online via one-step
    REINFORCE: each push's reward is the tracking error observed the NEXT time this is called for
    that env (i.e. after the push's effect has had time to show), minus a selectivity penalty on
    push magnitude (so it learns to go easy on pushes that wouldn't be disruptive anyway, rather
    than collapsing to always-maximal pushing)."""

    def __init__(self, num_envs: int, device: torch.device, selectivity_coef: float = 0.01, lr: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 12)).to(
            device
        )
        self.log_std = nn.Parameter(torch.zeros(6, device=device) - 1.0)  # starts at std=exp(-1)~=0.37
        self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=lr)
        self.selectivity_coef = selectivity_coef
        self.device = device

        # Detached bookkeeping buffers only -- deliberately NOT autograd-tracked. log_prob is
        # recomputed fresh (with grad) from these in finish_pending_and_update() instead of being
        # cached at act() time, because repeatedly writing a grad-tracked value into the same
        # persistent tensor via in-place indexing (`buf[ids] = value`) chains every past write
        # into one ever-growing graph through that buffer's whole history; backward() on a later
        # batch then tries to re-traverse (and re-free) nodes an earlier backward() already freed
        # ("Trying to backward through the graph a second time"). Recomputing from detached
        # (state, action) avoids this entirely -- each update's graph is fresh and independent.
        self.has_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pending_state = torch.zeros(num_envs, 7, device=device)
        self.pending_action = torch.zeros(num_envs, 6, device=device)
        self.reward_baseline = 0.0
        self._buffer_log_probs: list[torch.Tensor] = []
        self._buffer_rewards: list[torch.Tensor] = []
        self._updates_since_step = 0
        self.update_every = 8  # take an optimizer step roughly every 8 push events

    def _policy(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.net(state)[:, :6])  # (-1, 1), scaled to velocity_range by the caller
        std = torch.exp(self.log_std).clamp(min=1e-3).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def finish_pending_and_update(self, env_ids: torch.Tensor, error_now: torch.Tensor):
        """Credit-assign reward to whichever of env_ids had a pending push, and accumulate a
        REINFORCE sample (log_prob recomputed fresh from the stored detached state/action -- see
        __init__'s comment on why). Call this BEFORE overwriting pending_state/pending_action via
        act() for the same env_ids."""
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
        """Sample a new push action (-1, 1 per axis) for env_ids and remember it (detached) as
        pending, for finish_pending_and_update() to recompute log_prob from later."""
        with torch.no_grad():
            action = self._policy(state).sample()
        self.pending_state[env_ids] = state.detach()
        self.pending_action[env_ids] = action
        self.has_pending[env_ids] = True
        return action
