from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv




# Hamster specific imports

# maniskill stuff
import sapien
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

# === HAMSTER VLM client ===
from hamster_client_server import (
    HamsterVLMHTTP,
    get_camera_rgb_from_env,
    make_snapshot_path,
    save_annotated_image,
    save_local_annotated_from_sketch,
)

# === HAMSTER 2D->3D helpers ===
from hamster_2D_to_3D import (
    vlm_uv_to_world_points_from_env,
    extract_cube_goal_from_world_points,
    set_vlm_hl_path_on_env,    # optional (debug markers + HL path)
    inject_cube_estimate_p0,   # tiny convenience wrapper
    _has_points,
)


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "martin-soldani-the-university-of-sydney"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PullCubeHamster-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # my additions
    vlm_cam_name: str = "hamster_camera"

    # ---- NEW: P0 evaluation flags ----
    eval_obs_use_gt_cube: bool = True   # set False to replace cube with HAMSTER 3D estimate at eval
    eval_obs_use_gt_goal: bool = True   # keep True for P0 (goal stays GT)

    # HAMSTER INSTRUCTIONS
    instruction_hamster: str = "Push the blue cube to the goal area. Push the cube from its left side to the right."

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


def _get_first_base_env(obj):
    """
    Best-effort unwrapping to reach the first underlying (single) env instance
    that exposes `set_estimated_cube_from_vlm`.
    Works when num_eval_envs=1.
    """
    seen = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))

        # If this layer has the method, we’re done
        if hasattr(cur, "set_estimated_cube_from_vlm"):
            return cur

        # Common wrapper attributes to unwrap
        for name in ("unwrapped", "env", "_env", "venv"):
            nxt = getattr(cur, name, None)
            if nxt is not None:
                stack.append(nxt)

        # Vector envs: try first sub-env
        for name in ("envs", "_envs"):
            lst = getattr(cur, name, None)
            if isinstance(lst, (list, tuple)) and len(lst) > 0:
                stack.append(lst[0])
    raise RuntimeError("Could not find base env exposing set_estimated_cube_from_vlm. "
                       "Ensure num_eval_envs=1 and env=PullCubeHamster-v1.")


def _hide_all_markers(env):
    u = env.unwrapped
    try:
        import sapien
    except Exception:
        sapien = None

    def _offscreen_pose():
        # shove far below the table so it never appears
        return sapien.Pose(p=[10, 10, -10]) if sapien else None

    if getattr(u, "_cube_marker", None) is not None:
        if sapien: u._cube_marker.set_pose(_offscreen_pose())
    if getattr(u, "_goal_marker", None) is not None:
        if sapien: u._goal_marker.set_pose(_offscreen_pose())
    if getattr(u, "_waypoint_markers", None):
        for a in u._waypoint_markers:
            if sapien: a.set_pose(_offscreen_pose())


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    common_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        common_kwargs["control_mode"] = args.control_mode
        
    # WE MODIFY THIS PART TO ADD THE NEW ARGS
    # Train env uses full GT for P0
    train_kwargs = dict(**common_kwargs)
    train_kwargs.update(dict(
        obs_use_gt_cube=True,
        obs_use_gt_goal=True,
    ))

    # Eval env can switch GT off just for the cube (P0)
    eval_kwargs = dict(**common_kwargs)
    eval_kwargs.update(dict(
        obs_use_gt_cube=args.eval_obs_use_gt_cube,
        obs_use_gt_goal=args.eval_obs_use_gt_goal,
        ham_cam_name=args.vlm_cam_name,           # <- optional: override defaults
        # ham_cam_res=512,
        # ham_cam_fov_deg=55.0,
        # ham_cam_eye=(0.40,0.50,0.55),
        # ham_cam_target=(0.00,0.00,0.35),
    ))

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **train_kwargs
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs if not args.evaluate else 1,  # we’ll set 1 for P0
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **eval_kwargs
    )
        
    
    # WE KEEP THIS AS IS
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ADD NEW: a separate env for camera images if needed
    sensor_env = None
    if args.evaluate:
        # A separate, single env for grabbing camera images/depth
        sensor_kwargs = dict(obs_mode="rgbd", render_mode="rgb_array", sim_backend="physx_cuda")
        if args.control_mode is not None:
            sensor_kwargs["control_mode"] = args.control_mode
        sensor_env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=args.eval_reconfiguration_freq, **sensor_kwargs)
        # IMPORTANT: do NOT vector-wrap sensor_env. We want direct access to unwrapped.
        _ = sensor_env.reset()  # initialize sensors

    def _sync_objects_to_sensor_env(base_env, sensor_env):
        """
        Copy cube and goal poses from the real eval env to the sensor-tap env
        so the camera sees the same scene for this episode.
        """
        # Pull poses from the real eval env
        cube_pose = base_env.obj.pose  # Pose struct with .p (B,3) or (3,)
        goal_pose = base_env.goal_region.pose

        # Some builds store batched poses; select [0] if needed
        cube_p = cube_pose.p[0].detach().cpu().numpy() if getattr(cube_pose.p, "ndim", 1) == 2 else cube_pose.p
        goal_p = goal_pose.p[0].detach().cpu().numpy() if getattr(goal_pose.p, "ndim", 1) == 2 else goal_pose.p

        # Set in the sensor env
        sensor_env.unwrapped.obj.set_pose(Pose.create_from_pq(p=cube_p, q=[1,0,0,0]))
        sensor_env.unwrapped.goal_region.set_pose(Pose.create_from_pq(p=goal_p, q=euler2quat(0, np.pi/2, 0)))

    # === HAMSTER VLM client helper ===
    vlm_client = None
    snap_root = None
    if args.evaluate:
        vlm_client = HamsterVLMHTTP()  # or HamsterVLMHTTP(base_url="http://127.0.0.1:8000")
        # snapshot folder under the run dir
        run_dir = f"{os.path.dirname(args.checkpoint)}"
        snap_root = os.path.join(run_dir, "snapshots")

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**train_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**eval_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # === HAMSTER VLM calls - episode counter ===
    # how many episodes do we intend to evaluate this pass?
    horizon = max_episode_steps  # you already computed this earlier
    target_eps = args.num_eval_steps // horizon
    # If you ever want to allow a partial last episode, use ceil instead:
    # target_eps = math.ceil(args.num_eval_steps / horizon)

    injections_done = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)

    # ---- NEW: P0 injection at reset ---- this worked before 
    # if (not args.eval_obs_use_gt_cube) and (args.num_eval_envs == 1):
    #     base_eval_env = _get_first_base_env(eval_envs)
    #     # TODO: replace the next line with YOUR real HAMSTER → 3D estimate (torch.Tensor shape [3] or [1,3])
    #     cube_3d_est = torch.as_tensor([1.0, 1.0, base_eval_env.cube_half_size], dtype=torch.float32, device="cpu")
    #     # If you already have (x,y,z) from HAMSTER, do: cube_3d_est = torch.tensor([x, y, z], dtype=torch.float32)
    #     base_eval_env.set_estimated_cube_from_vlm(cube_3d_est)

    # TEST: COMMENT OUT FOR NOW AS TOO MANY HAMSTER CALLS OTHERWISE
    # # ---- HAMSTER injection (P0) after first eval reset ----
    # if (not args.eval_obs_use_gt_cube) and (args.num_eval_envs == 1) and (sensor_env is not None):
    #     base_eval_env = _get_first_base_env(eval_envs)

    #     # 1) Mirror the episode scene into the sensor tap (NO reset afterwards)
    #     _sync_objects_to_sensor_env(base_eval_env, sensor_env)

    #     # 2) Grab a frame from base_camera_1 and save it
    #     rgb = get_camera_rgb_from_env(sensor_env, cam_name=args.vlm_cam_name)
    #     snap_path = make_snapshot_path(snap_root, prefix="ep_start")

    #     # 3) Ask Hamster for the 2D path using THIS snapshot; the client saves exactly what it sends
    #     instruction = args.instruction_hamster  # TODO: make CLI arg if desired
    #     sketch = vlm_client.get_path(
    #         rgb,
    #         instruction,
    #         resize_to=(512, 512),   # must match what you’ll use for back-projection
    #         save_to=snap_path,      # .../snapshots/ep_start_YYYYMMDD_HHMMSS.png
    #         return_image=True,
    #     )
    #     uv_coords = [(wp.u, wp.v) for wp in sketch.waypoints]
    #     vlm_size = tuple(sketch.meta["vlm_image_size"])  # (W,H), e.g. (512,512)

    #     if len(uv_coords) == 0:
    #         print("[VLM] No waypoints returned; skipping injection this episode.")
    #     else:
    #         # 4) Convert UV->3D using the SAME camera frame you mirrored
    #         #    (Fetch the rich obs dict WITHOUT resetting.)
    #         obs_dict = (
    #             getattr(sensor_env.unwrapped, "get_obs", None)
    #             or getattr(sensor_env.unwrapped, "get_observation", None)
    #             or getattr(sensor_env.unwrapped, "_get_obs", None)
    #         )()
    #         pts_3d = vlm_uv_to_world_points_from_env(
    #             obs=obs_dict,
    #             uv_coords_norm=uv_coords,
    #             vlm_image_size=vlm_size,      # MUST match the size sent to Hamster
    #             cam_name=args.vlm_cam_name,
    #         )

    #         # 5) Extract endpoints and inject cube estimate (P0)
    #         tcp_world = base_eval_env.agent.tcp.pose.p[0].detach().cpu().numpy()
    #         cube_pos, goal_pos = extract_cube_goal_from_world_points(
    #             pts_3d, tcp_world=tcp_world, endpoint_strategy="first_last"
    #         )
    #         if cube_pos is not None:
    #             inject_cube_estimate_p0(base_eval_env, cube_pos)
    #             base_eval_env.unwrapped.show_cube_marker(cube_pos)  # marker moves to estimate
    #         if goal_pos is not None:
    #             base_eval_env.unwrapped.show_goal_marker(goal_pos)
    #         if _has_points(pts_3d):
    #             base_eval_env.unwrapped.set_waypoint_markers(pts_3d)

    #     # (Optional) save annotated image if server returned one
    #     anno_path = snap_path.replace(".png", "_annotated.png")
    #     ok = save_local_annotated_from_sketch(
    #         snapshot_path=snap_path,   # this is exactly what we sent (512x512)
    #         sketch=sketch,             # contains the waypoints we parsed
    #         out_path=anno_path,
    #         quest=instruction,         # optional label in the top-left corner
    #     )
    #     if not ok:
    #         print("[VLM] No annotated image could be generated (no waypoints or missing snapshot).")





    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()




            # ---- NEW: P0 injection each eval cycle ---- workeed beofre
            # if (not args.eval_obs_use_gt_cube) and (args.num_eval_envs == 1):
            #     base_eval_env = _get_first_base_env(eval_envs)
            #     # TODO: replace with your real HAMSTER → 3D estimate
            #     cube_3d_est = torch.as_tensor([1.0, 1.0, base_eval_env.cube_half_size], dtype=torch.float32, device="cpu")
            #     base_eval_env.set_estimated_cube_from_vlm(cube_3d_est)

            # ---- HAMSTER injection (P0) each periodic eval reset ----
            if (not args.eval_obs_use_gt_cube) and (args.num_eval_envs == 1) and (sensor_env is not None):
                base_eval_env = _get_first_base_env(eval_envs)

                # 1) Mirror the episode scene into the sensor tap (NO reset afterwards)
                _sync_objects_to_sensor_env(base_eval_env, sensor_env)
                _hide_all_markers(sensor_env)

                # 2) Grab a frame from base_camera_1 and save it
                rgb = get_camera_rgb_from_env(sensor_env, cam_name=args.vlm_cam_name)
                snap_path = make_snapshot_path(snap_root, prefix="ep_start")

                # 3) Ask Hamster for the 2D path using THIS snapshot; the client saves exactly what it sends
                instruction = args.instruction_hamster  # TODO: make CLI arg if desired
                sketch = vlm_client.get_path(
                    rgb,
                    instruction,
                    resize_to=(512, 512),   # must match what you’ll use for back-projection
                    save_to=snap_path,      # .../snapshots/ep_start_YYYYMMDD_HHMMSS.png
                    return_image=True,
                )
                uv_coords = [(wp.u, wp.v) for wp in sketch.waypoints]
                vlm_size = tuple(sketch.meta["vlm_image_size"])  # (W,H), e.g. (512,512)

                if len(uv_coords) == 0:
                    print("[VLM] No waypoints returned; skipping injection this episode.")
                else:
                    # 4) Convert UV->3D using the SAME camera frame you mirrored
                    #    (Fetch the rich obs dict WITHOUT resetting.)
                    obs_dict = (
                        getattr(sensor_env.unwrapped, "get_obs", None)
                        or getattr(sensor_env.unwrapped, "get_observation", None)
                        or getattr(sensor_env.unwrapped, "_get_obs", None)
                    )()
                    pts_3d = vlm_uv_to_world_points_from_env(
                        obs=obs_dict,
                        uv_coords_norm=uv_coords,
                        vlm_image_size=vlm_size,      # MUST match the size sent to Hamster
                        cam_name=args.vlm_cam_name,
                    )

                    # 5) Extract endpoints and inject cube estimate (P0)
                    tcp_world = base_eval_env.agent.tcp.pose.p[0].detach().cpu().numpy()
                    cube_pos, goal_pos = extract_cube_goal_from_world_points(
                        pts_3d, tcp_world=tcp_world, endpoint_strategy="first_last"
                    )
                    if cube_pos is not None:
                        inject_cube_estimate_p0(base_eval_env, cube_pos)
                        base_eval_env.unwrapped.show_cube_marker(cube_pos)  # marker moves to estimate
                    if goal_pos is not None:
                        base_eval_env.unwrapped.show_goal_marker(goal_pos)
                    if _has_points(pts_3d):
                        base_eval_env.unwrapped.set_waypoint_markers(pts_3d)

                # (Optional) save annotated image if server returned one
                anno_path = snap_path.replace(".png", "_annotated.png")
                ok = save_local_annotated_from_sketch(
                    snapshot_path=snap_path,   # this is exactly what we sent (512x512)
                    sketch=sketch,             # contains the waypoints we parsed
                    out_path=anno_path,
                    quest=instruction,         # optional label in the top-left corner
                )
                if not ok:
                    print("[VLM] No annotated image could be generated (no waypoints or missing snapshot).")

                injections_done += 1

                            


            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    # debug print
                    if args.evaluate:
                        tcp_xy = base_eval_env.unwrapped.agent.tcp.pose.p[0, :2].cpu().numpy()
                        goal_xy = base_eval_env.unwrapped.goal_region.pose.p[0, :2].cpu().numpy()
                        print("tcp→goal dist @t0:", np.linalg.norm(tcp_xy - goal_xy))

                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

                        if injections_done < target_eps:
                            # # comment out for now as too many calls
                            # Re-inject a new HAMSTER estimate for the next episode (num_eval_envs==1 in P0)
                            if (not args.eval_obs_use_gt_cube) and (args.num_eval_envs == 1) and (sensor_env is not None):
                                base_eval_env = _get_first_base_env(eval_envs)

                                # 1) Mirror the episode scene into the sensor tap (NO reset afterwards)
                                _sync_objects_to_sensor_env(base_eval_env, sensor_env)
                                _hide_all_markers(sensor_env)

                                # 2) Grab a frame from base_camera_1 and save it
                                rgb = get_camera_rgb_from_env(sensor_env, cam_name=args.vlm_cam_name)
                                snap_path = make_snapshot_path(snap_root, prefix="ep_start")

                                # 3) Ask Hamster for the 2D path using THIS snapshot; the client saves exactly what it sends
                                instruction = args.instruction_hamster  # TODO: make CLI arg if desired
                                sketch = vlm_client.get_path(
                                    rgb,
                                    instruction,
                                    resize_to=(512, 512),   # must match what you’ll use for back-projection
                                    save_to=snap_path,      # .../snapshots/ep_start_YYYYMMDD_HHMMSS.png
                                    return_image=True,
                                )
                                uv_coords = [(wp.u, wp.v) for wp in sketch.waypoints]
                                vlm_size = tuple(sketch.meta["vlm_image_size"])  # (W,H), e.g. (512,512)

                                if len(uv_coords) == 0:
                                    print("[VLM] No waypoints returned; skipping injection this episode.")
                                else:
                                    # 4) Convert UV->3D using the SAME camera frame you mirrored
                                    #    (Fetch the rich obs dict WITHOUT resetting.)
                                    obs_dict = (
                                        getattr(sensor_env.unwrapped, "get_obs", None)
                                        or getattr(sensor_env.unwrapped, "get_observation", None)
                                        or getattr(sensor_env.unwrapped, "_get_obs", None)
                                    )()
                                    pts_3d = vlm_uv_to_world_points_from_env(
                                        obs=obs_dict,
                                        uv_coords_norm=uv_coords,
                                        vlm_image_size=vlm_size,      # MUST match the size sent to Hamster
                                        cam_name=args.vlm_cam_name,
                                    )

                                    # 5) Extract endpoints and inject cube estimate (P0)
                                    tcp_world = base_eval_env.agent.tcp.pose.p[0].detach().cpu().numpy()
                                    cube_pos, goal_pos = extract_cube_goal_from_world_points(
                                        pts_3d, tcp_world=tcp_world, endpoint_strategy="first_last"
                                    )
                                    if cube_pos is not None:
                                        inject_cube_estimate_p0(base_eval_env, cube_pos)
                                        base_eval_env.unwrapped.show_cube_marker(cube_pos)  # marker moves to estimate
                                    if goal_pos is not None:
                                        base_eval_env.unwrapped.show_goal_marker(goal_pos)
                                    if _has_points(pts_3d):
                                        base_eval_env.unwrapped.set_waypoint_markers(pts_3d)

                                # (Optional) save annotated image if server returned one
                                anno_path = snap_path.replace(".png", "_annotated.png")
                                ok = save_local_annotated_from_sketch(
                                    snapshot_path=snap_path,   # this is exactly what we sent (512x512)
                                    sketch=sketch,             # contains the waypoints we parsed
                                    out_path=anno_path,
                                    quest=instruction,         # optional label in the top-left corner
                                )
                                if not ok:
                                    print("[VLM] No annotated image could be generated (no waypoints or missing snapshot).")
                                
                                injections_done += 1


            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
 