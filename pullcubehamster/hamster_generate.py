# hamster_generate.py
import numpy as np
import torch
from typing import Iterable, List, Optional, Sequence, Tuple

# Make sure this import finds your file (add repo root to PYTHONPATH if needed)
from vision_utils import uv_norm_to_world_points


# ---------- Camera / depth utilities ----------

def get_depth_K_cam2world_from_obs(
    obs: dict,
    cam_name: str = "base_camera_1",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract depth (meters), K (3x3, OpenCV), cam2world_gl (4x4) from ManiSkill obs dict.
    Handles torch/numpy and singleton batch dims.
    """
    # Depth → meters with NaN for invalid
    depth = obs["sensor_data"][cam_name]["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = depth.squeeze()
    depth = depth.astype(np.float32)
    depth[depth <= 0] = np.nan
    # If depth is in mm in your setup, uncomment:
    depth_m = depth / 1000.0  # mm → m

    # Intrinsics / extrinsics
    K = obs["sensor_param"][cam_name]["intrinsic_cv"]
    cam2world = obs["sensor_param"][cam_name]["cam2world_gl"]
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(cam2world, torch.Tensor):
        cam2world = cam2world.detach().cpu().numpy()
    if K.ndim == 3 and K.shape[0] == 1:
        K = K[0]
    if cam2world.ndim == 3 and cam2world.shape[0] == 1:
        cam2world = cam2world[0]

    return depth_m, K, cam2world


def vlm_uv_to_world_points_from_obs(
    obs: dict,
    uv_coords_norm: Sequence[Tuple[float, float]],
    vlm_image_size: Tuple[int, int] = (512, 512),
    cam_name: str = "base_camera_1",
) -> np.ndarray:
    """
    Convert a list of normalized UV coords (0..1) from the VLM into Nx3 world points (z-up).
    Returns a numpy array of shape (N,3). Filters invalid points.
    """
    depth_m, K, cam2world = get_depth_K_cam2world_from_obs(obs, cam_name=cam_name)
    pts_list = uv_norm_to_world_points(
        uv_list_norm=uv_coords_norm,
        depth_m=depth_m,
        K_cv=K,
        cam2world_gl=cam2world,
        vlm_image_size=vlm_image_size,
    )
    if len(pts_list) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(pts_list, axis=0).astype(np.float32)  # (N,3)


# ---------- Cube / goal extraction ----------

def extract_cube_goal_from_world_points(
    points_3d: np.ndarray,
    tcp_world: Optional[Sequence[float]] = None,
    prior_cube: Optional[Sequence[float]] = None,
    prior_goal: Optional[Sequence[float]] = None,
    table_snap_z: Optional[float] = None,
    endpoint_strategy: str = "auto",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Given a polyline of 3D points (N,3), return (cube_pos, goal_pos) as torch.float32 (3,)
    using robust heuristics:

    - If N == 0: returns (None, None)
    - If N == 1: treat as cube only -> (cube, None)
    - If N >= 2:
        * endpoint_strategy='auto' (default):
            - prefer the endpoint closer to tcp_world as cube (if tcp_world provided)
            - else prefer closer to prior_cube (if provided)
            - else fall back to ordered endpoints: first -> cube, last -> goal
        * endpoint_strategy='first_last': first -> cube, last -> goal

    Options:
    - table_snap_z: if provided, set z to this for both outputs (useful for stability)
    - prior_*: used as tie-breakers or when tcp_world is not available
    """
    if points_3d is None or len(points_3d) == 0:
        return None, None

    pts = np.asarray(points_3d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_3d must be (N,3); got {pts.shape}")

    if pts.shape[0] == 1:
        cube = pts[0].copy()
        if table_snap_z is not None:
            cube[2] = table_snap_z
        return torch.as_tensor(cube, dtype=torch.float32), None

    # Two endpoints
    a, b = pts[0], pts[-1]

    # Heuristic selection
    if endpoint_strategy == "first_last":
        cube, goal = a.copy(), b.copy()
    else:
        # auto: use tcp or priors when available
        def dist(p, q):
            return np.linalg.norm(np.asarray(p) - np.asarray(q))

        if tcp_world is not None:
            cube = a if dist(a, tcp_world) <= dist(b, tcp_world) else b
            goal = b if cube is a else a  # the other one
        elif prior_cube is not None:
            cube = a if dist(a, prior_cube) <= dist(b, prior_cube) else b
            goal = b if cube is a else a
        elif prior_goal is not None:
            goal = a if dist(a, prior_goal) <= dist(b, prior_goal) else b
            cube = b if goal is a else a
        else:
            # fall back: assume input order is cube→goal
            cube, goal = a.copy(), b.copy()

    # Optional Z snapping
    if table_snap_z is not None:
        cube = np.array([cube[0], cube[1], table_snap_z], dtype=np.float32)
        goal = np.array([goal[0], goal[1], table_snap_z], dtype=np.float32)

    return torch.as_tensor(cube, dtype=torch.float32), torch.as_tensor(goal, dtype=torch.float32)


# ---------- Env helpers (optional, for convenience) ----------

def set_vlm_hl_path_on_env(
    env,
    points_3d: np.ndarray,
    visualize_markers: bool = False,
):
    """
    Optionally set the HL path and add debug markers. No-ops if points_3d is empty.
    """
    if points_3d is None or len(points_3d) == 0:
        return
    hl_path = torch.as_tensor(points_3d, dtype=torch.float32)
    env.unwrapped.set_high_level_path(hl_path)
    if visualize_markers:
        for p in points_3d:
            env.unwrapped.add_visual_marker_ball(position=p.tolist())


def inject_cube_estimate_p0(env, cube_pos_3d: torch.Tensor):
    """
    P0-only: inject the cube estimate into PullCubeHamster-v1 so obs uses it.
    Accepts (3,) or (1,3). Env will broadcast to batch internally.
    """
    env.unwrapped.set_estimated_cube_from_vlm(cube_pos_3d)
    print("Injected cube estimate into env for P0:", cube_pos_3d.tolist())
