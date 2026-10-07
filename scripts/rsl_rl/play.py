"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--clean_vis", action="store_true", default=False,
    help="Hide the per-body reference markers (keep only the anchor/root marker) for a less "
         "cluttered demo video -- see MotionCommandCfg.show_body_markers.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import collections
import gymnasium as gym
import json
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.eval_metrics import compute_settling_metrics
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        # NOTE: this was previously only done in the --wandb_path branch above, so evaluating a
        # locally-loaded checkpoint (--load_run/--checkpoint) with --motion_file left
        # env_cfg.commands.motion.motion_file unset -> "Missing values ... motion_file" on
        # gym.make(). --motion_file is the only way to set it in this branch (no wandb artifact
        # to fall back on), so always apply it here.
        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

    if args_cli.clean_vis:
        env_cfg.commands.motion.show_body_markers = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # Optional: save tracking-error / success-rate metrics to a JSON file for a fixed number
    # of steps, for the A/B/C kit_test/amass_test comparison (Windows tutorial, step 11).
    # Without this, the play loop below runs forever (`while simulation_app.is_running()`),
    # so batch eval scripts (04_eval.ps1) that call play.py once per clip need a concrete
    # stopping point -- EVAL_OUT set means "run EVAL_STEPS steps, then save and exit" instead
    # of "run until the window is closed".
    eval_out = os.environ.get("EVAL_OUT")
    eval_steps = int(os.environ.get("EVAL_STEPS", "1500"))  # 1500 steps @ 50 Hz control = 30s
    # ManagerBasedRLEnv.step() calls termination_manager.compute() BEFORE command_manager.compute()
    # (see envs/manager_based_rl_env.py) -- on a freshly-created env, the very first step() checks
    # termination against MotionCommand.body_pos_relative_w while it's still zero-initialized
    # (command_manager.reset(), called during env setup, only resamples the clip/robot pose; it does
    # NOT populate body_pos_relative_w -- that only happens inside _update_command(), called from
    # command_manager.compute(), which runs LATER in that same first step()). So every eval run gets
    # exactly one guaranteed spurious termination on its first recorded step. With only ~3
    # episode-lengths fitting in a 1500-step eval window, that single bogus failure caps measured
    # success_rate at ~2/3 regardless of actual tracking quality -- confirmed by the exact 0.667
    # clustering across clips with wildly different error_body_pos. error_body_pos/error_joint_pos/
    # error_anchor_pos are NOT affected by this specific bug: play.py reads motion_term.metrics
    # AFTER env.step() returns, by which point command_manager.compute() has already refreshed
    # body_pos_relative_w for that same step (termination fires first, using the old/zero value, but
    # the logged error always reflects the just-recomputed one). success_rate is a count over only
    # ~3 discrete episode-trials, so losing 1 to this bug is catastrophic there specifically. See
    # docs/진행상황_연구노트.md.
    EVAL_WARMUP_STEPS = 2
    if eval_out:
        motion_term = env.unwrapped.command_manager.get_term("motion")
        termination_manager = env.unwrapped.termination_manager
        eval_log = collections.defaultdict(list)
        eval_fails = 0
        eval_timeouts = 0

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if eval_out:
            for key in ("error_body_pos", "error_joint_pos", "error_anchor_pos"):
                eval_log[key].append(motion_term.metrics[key].mean().item())
            # Skip the warmup window for success_rate counting (see EVAL_WARMUP_STEPS comment
            # above) -- the stale-buffer artifact causes a guaranteed spurious termination on
            # step 0 of every eval run, which with only ~3 episodes fitting in the eval window
            # caps measured success_rate at ~2/3 regardless of actual tracking quality.
            if timestep >= EVAL_WARMUP_STEPS:
                eval_fails += int(termination_manager.terminated.sum().item())
                eval_timeouts += int(termination_manager.time_outs.sum().item())
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        elif eval_out:
            timestep += 1
            if timestep >= eval_steps:
                break

    if eval_out:
        # CommandTerm.compute() calls _update_metrics() BEFORE _update_command() (see
        # managers/command_manager.py), so eval_log's error_* values are also one step
        # "behind" the reference buffer -- on the very first recorded step this means they're
        # computed against the same zero-initialized body_pos_relative_w as the termination
        # check above, not just reflecting last step's (otherwise negligible) lag. Drop the
        # same EVAL_WARMUP_STEPS from these means too, for the same reason as eval_fails/
        # eval_timeouts above. eval_log itself is left untrimmed (full warmup included) since
        # compute_settling_metrics() below does its own warmup_steps trim and offset math.
        results = {key: sum(vals[EVAL_WARMUP_STEPS:]) / len(vals[EVAL_WARMUP_STEPS:]) for key, vals in eval_log.items()}
        results["success_rate"] = eval_timeouts / max(eval_fails + eval_timeouts, 1)
        results["eval_steps"] = eval_steps
        # Settling time / overshoot / steady-state error, adapted from step-response control
        # analysis to this continuous-tracking task (see eval_metrics.py docstring) -- computed
        # from the initial transient only, since that's the one synchronized "step" every env in
        # this run shares (all envs reset together right before the loop above starts).
        results.update(compute_settling_metrics(eval_log["error_body_pos"], dt=env.unwrapped.step_dt))
        with open(eval_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved eval metrics to {eval_out}: {results}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
