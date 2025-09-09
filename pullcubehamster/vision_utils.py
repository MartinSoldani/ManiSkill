import torch
import numpy as np
import numpy as np
import torch

def uv_norm_to_world_points(
    uv_list_norm,
    depth_m,            # (H, W) meters, NaN invalid
    K_cv,               # (3,3) intrinsic_cv (OpenCV)
    cam2world_gl,       # (4,4) cam->world in OpenGL camera frame
    vlm_image_size      # (W, H) that the VLM used (e.g., (512, 512))
):
    """
    Convert normalized 2D coords (0..1) from the VLM to 3D world (z-up).

    Steps:
      1) (xn,yn) -> pixel (u,v) using the *exact* VLM image size
      2) Back-project to CV camera frame using K_cv and depth (meters)
      3) Convert CV camera point -> GL camera point (flip y,z axes)
      4) Transform with cam2world_gl -> world point (z-up)

    Returns: list of np.array([x,y,z]) in world frame
    """
    # to numpy
    if isinstance(depth_m, torch.Tensor): depth_m = depth_m.detach().cpu().numpy()
    if isinstance(K_cv, torch.Tensor):    K_cv    = K_cv.detach().cpu().numpy()
    if isinstance(cam2world_gl, torch.Tensor):
        cam2world_gl = cam2world_gl.detach().cpu().numpy()

    W_vlm, H_vlm = vlm_image_size  # e.g., (512, 512)

    # If your depth map is the same resolution as the RGB sent to the VLM,
    # H,W should match (otherwise you'd need to resample depth or scale pixels).
    H_d, W_d = depth_m.shape[:2]

    fx, fy = K_cv[0,0], K_cv[1,1]
    cx, cy = K_cv[0,2], K_cv[1,2]

    # CV -> GL conversion (flip y and z)
    T_gl_from_cv = np.diag([1.0, -1.0, -1.0, 1.0])

    points_world = []
    for xn, yn in uv_list_norm:
        # 1) normalized -> pixel in the *VLM image*
        u_vlm = int(round(xn * (W_vlm - 1)))
        v_vlm = int(round(yn * (H_vlm - 1)))
        u_vlm = np.clip(u_vlm, 0, W_vlm - 1)
        v_vlm = np.clip(v_vlm, 0, H_vlm - 1)

        # If your depth map is the same resolution as the VLM input, this is fine.
        # If not, map (u_vlm, v_vlm) to the depth resolution here.
        if (W_vlm, H_vlm) != (W_d, H_d):
            u = int(round(u_vlm * (W_d / W_vlm)))
            v = int(round(v_vlm * (H_d / H_vlm)))
            u = np.clip(u, 0, W_d - 1)
            v = np.clip(v, 0, H_d - 1)
        else:
            u, v = u_vlm, v_vlm

        # 2) depth at pixel (meters, forward CV +z)
        z = depth_m[v, u]
        if not np.isfinite(z) or z <= 0:
            # optional fallback: small median neighborhood
            win = depth_m[max(0,v-2):v+3, max(0,u-2):u+3]
            z = np.nanmedian(win)
            if not np.isfinite(z) or z <= 0:
                continue

        x_cam_cv = (u - cx) * z / fx
        y_cam_cv = (v - cy) * z / fy
        p_cam_cv = np.array([x_cam_cv, y_cam_cv, z, 1.0], dtype=np.float32)

        # 3) CV camera -> GL camera
        p_cam_gl = T_gl_from_cv @ p_cam_cv

        # 4) GL camera -> world (z-up)
        p_world_h = cam2world_gl @ p_cam_gl
        points_world.append(p_world_h[:3])

    return points_world


