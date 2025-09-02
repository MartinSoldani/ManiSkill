# play.py
import torch
import os
import imageio
import numpy as np
import gymnasium as gym
import mani_skill.envs  # register envs
from mani_skill.utils.wrappers import CPUGymWrapper

from vision_utils import *

# Set the environment and wrap it to get CPU numpy obs
env = gym.make("PullCubeHL-v1", obs_mode="rgbd", render_mode="human")  # use "human" to show GUI
env = CPUGymWrapper(env)

# Reset and get initial obs
obs, _ = env.reset(seed=42)

print("Cube pos:", env.unwrapped.obj.pose.p)
print("Goal pos:", env.unwrapped.goal_region.pose.p)


# === PROCESS DEPTH ===
cam_name = "base_camera_1"
depth = obs["sensor_data"][cam_name]["depth"]
if isinstance(depth, torch.Tensor):
    depth = depth.cpu().numpy()
depth = depth.squeeze()
depth = depth.astype(np.float32)
depth[depth <= 0] = np.nan
depth /= 1000.0  # mm → meters

# === CAMERA POSE ===
cam_pose = obs["sensor_param"][cam_name]["cam2world_gl"]
if isinstance(cam_pose, torch.Tensor):
    cam_pose = cam_pose.squeeze(0).cpu().numpy()
print("Camera world pose:", cam_pose)


# === CAMERA PARAMS ===
K = obs["sensor_param"][cam_name]["intrinsic_cv"]
cam2world = obs["sensor_param"][cam_name]["cam2world_gl"]
if isinstance(K, torch.Tensor):
    K = K.cpu().numpy()
if isinstance(cam2world, torch.Tensor):
    cam2world = cam2world.cpu().numpy()
if K.ndim == 3 and K.shape[0] == 1:
    K = K[0]
if cam2world.ndim == 3 and cam2world.shape[0] == 1:
    cam2world = cam2world[0]

# === HL-VLM OUTPUT ===
uv_coords = [(0.50, 0.53), (0.59, 0.48)]
rgb = obs["sensor_data"][cam_name]["rgb"]      # to confirm H,W if needed
depth_m = depth                                # your processed meters/NaN map
points_3d = uv_norm_to_world_points(
    uv_coords,          # [(0.50, 0.53), (0.59, 0.48)]
    depth_m,
    K,                  # intrinsic_cv
    cam2world,          # cam2world_gl from obs
    (512, 512)          # <-- the exact size the VLM used
)

print("3D Points:", points_3d)

# === ADD BALLS AT THOSE POINTS ===
for point in points_3d:
    env.unwrapped.add_visual_marker_ball(position=point)

# === RUN SIMULATION FOR A FEW STEPS TO TRIGGER RENDERING ===
for _ in range(300):  # let simulation render and stabilize
    action = env.action_space.sample() * 0  # or np.zeros_like(...)
    obs, _, _, _, _ = env.step(action)
    env.render()  # required to update GUI and visuals

# === SAVE IMAGES AFTER VISUAL MARKERS APPEAR ===
save_dir = "hl_images"
os.makedirs(save_dir, exist_ok=True)

for cam_name, cam_data in obs["sensor_data"].items():
    rgb = cam_data["rgb"]
    imageio.imwrite(f"{save_dir}/{cam_name}.png", rgb)

print("Saved camera images to:", save_dir)
