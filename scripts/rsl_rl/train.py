# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--registry_name", type=str, default=None, help="The name of the wandb registry (single motion; smoke test)."
)
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help=(
        "Directory of local motion .npz files to train on all of them at once (multi-clip; used for the"
        " A/B/C AMASS->KIT conditions). Overrides --registry_name if both are given."
    ),
)
parser.add_argument(
    "--kl_coef",
    type=float,
    default=0.0,
    help=(
        "RL's Razor-style KL regularization coefficient (Shenfeld et al., 2025, arXiv:2509.04259): adds"
        " kl_coef * KL(pi_new || pi_reference) to the PPO loss, where pi_reference is a frozen copy of the"
        " --load_run/--checkpoint being resumed from. 0 (default) disables it. Requires --resume True; meant"
        " for condition B (KIT fine-tuning from A's checkpoint) to reduce seed-to-seed variance and forgetting."
    ),
)
parser.add_argument(
    "--adaptive_clip_sampling",
    action="store_true",
    default=False,
    help=(
        "D2 (GMT/PHC-style, opt-in): with --motion_dir (multiple clips), sample clips proportional to"
        " their recent EMA'd failure rate instead of uniformly, so the policy gets more practice on"
        " clips it currently fails. Off by default; does not affect A/B/C/R1/D1/R2 runs. See"
        " whole_body_tracking/tasks/tracking/mdp/commands.py's MotionCommand._adaptive_clip_sampling."
    ),
)
parser.add_argument(
    "--segment_adaptive_sampling",
    action="store_true",
    default=False,
    help=(
        "N1 (Stubborn-style, arXiv:2606.12814): with --motion_dir (multiple clips), keep clip"
        " selection uniform but bias the start frame within whichever clip is picked toward that"
        " clip's own recently-failing segment. Fixes D2's failure mode (a single infeasible clip"
        " swallowing most of the sampling budget) by never letting clip choice itself be"
        " failure-weighted. Off by default. Mutually exclusive with --adaptive_clip_sampling (D2);"
        " if both are given, D2 takes precedence. See"
        " whole_body_tracking/tasks/tracking/mdp/commands.py's MotionCommand._segment_adaptive_clip_sampling."
    ),
)
parser.add_argument(
    "--r2_curriculum",
    action="store_true",
    default=False,
    help=(
        "Enable the R2 curriculum (KungfuBot-style, Xie et al., 2025, arXiv:2506.12851): adaptive"
        " exp-reward tracking std (sigma <- min(sigma, EMA(error))) plus exponential termination-threshold"
        " and penalty-weight curricula. See whole_body_tracking/utils/r2_curriculum.py. Off by default;"
        " does not affect A/B/C/R1/D1 runs."
    ),
)
parser.add_argument(
    "--symmetry_loss",
    type=float,
    default=0.0,
    help=(
        "N2 candidate (Mittal et al., 2024, arXiv:2403.04359): left-right mirror-consistency"
        " auxiliary loss, coefficient * MSE(pi(mirror(obs)), mirror(pi(obs))). 0 (default) disables"
        " it. See whole_body_tracking/utils/symmetry.py for the G1-specific mirror transform."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.registry_name and not args_cli.motion_dir:
    parser.error("one of --registry_name or --motion_dir is required")
if args_cli.kl_coef > 0 and not args_cli.resume:
    parser.error("--kl_coef > 0 requires --resume True (it regularizes toward the --load_run/--checkpoint being resumed from)")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.kl_regularized_ppo import attach_kl_regularization
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from whole_body_tracking.utils.r2_curriculum import attach_r2_curriculum
from whole_body_tracking.utils.symmetry import g1_tracking_mirror_augmentation

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # load the motion file(s): either a local directory of npz clips (multi-clip A/B/C
    # training, --motion_dir) or a single clip from the wandb registry (--registry_name,
    # original single-motion / smoke-test path)
    import pathlib

    registry_name = args_cli.registry_name
    if args_cli.motion_dir:
        motion_dir = pathlib.Path(args_cli.motion_dir)
        motion_files = sorted(str(p) for p in motion_dir.glob("*.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No .npz files found in --motion_dir {motion_dir}")
        print(f"[INFO] Training on {len(motion_files)} motion clips from {motion_dir}")
        env_cfg.commands.motion.motion_file = motion_files
        if args_cli.adaptive_clip_sampling:
            env_cfg.commands.motion.adaptive_clip_sampling = True
            print("[INFO] D2 adaptive clip sampling enabled (samples clips proportional to EMA'd failure rate)")
        if args_cli.segment_adaptive_sampling:
            env_cfg.commands.motion.segment_adaptive_sampling = True
            print("[INFO] N1 segment-adaptive sampling enabled (uniform clip choice, failure-biased start segment)")
        # not pulled from the wandb registry, so there's nothing to link as a used artifact
        registry_name = None
    else:
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    if args_cli.r2_curriculum:
        attach_r2_curriculum(env_cfg)

    if args_cli.symmetry_loss > 0:
        agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=args_cli.symmetry_loss,
            data_augmentation_func=g1_tracking_mirror_augmentation,
        )
        print(f"[INFO] N2 symmetry auxiliary loss enabled (Mittal et al., 2024): coeff={args_cli.symmetry_loss}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # optionally regularize fine-tuning toward this same checkpoint (RL's Razor-style
        # KL penalty, see whole_body_tracking/utils/kl_regularized_ppo.py)
        if args_cli.kl_coef > 0:
            attach_kl_regularization(runner, resume_path, args_cli.kl_coef)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
