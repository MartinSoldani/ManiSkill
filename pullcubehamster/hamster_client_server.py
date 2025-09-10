# hamster_client_server.py
from __future__ import annotations
import os, re, time, base64, pathlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image as PILImage

# ============================== Types ==============================

GRIPPER_CLOSE, GRIPPER_OPEN = 0, 1

@dataclass
class Waypoint2D:
    u: float
    v: float
    grip: Optional[int] = None  # 0=close, 1=open, None=unspecified

@dataclass
class VLMSketch:
    waypoints: List[Waypoint2D]
    meta: Dict[str, Any]

# ============================== Utils ==============================

DEFAULT_MODEL = "HAMSTER_dev"
DEFAULT_PORT = 8000
IP_FILE_NAME = "ip_eth0.txt"

def _to_hwc_uint8(arr) -> np.ndarray:
    """
    Convert many common camera array layouts to uint8 HxWx3 (RGB).
    Handles:
      - torch tensors / numpy arrays
      - channel-first (3,H,W), (1,3,H,W)
      - channels-last with batch dims (1,H,W,3), (1,1,W,3), etc.
      - floats in [0,1] or [0,255]
    Selects the first item along any leading non-(H,W,C) axes.
    """
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(arr)

    # If there is a channel axis of length 3 anywhere, move it to the end
    axes_with_3 = [i for i, d in enumerate(x.shape) if d == 3]
    if axes_with_3 and axes_with_3[-1] != x.ndim - 1:
        x = np.moveaxis(x, axes_with_3[-1], -1)  # put that '3' as last axis

    # Reduce any remaining leading dims until we have at most 3 dims
    while x.ndim > 3:
        x = x[0]

    # Now handle channel-first 3D: (3,H,W) -> (H,W,3)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.moveaxis(x, 0, -1)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 after normalization, got shape {x.shape}")

    # dtype to uint8
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating):
            m = float(np.nanmax(x)) if x.size else 1.0
            if m <= 1.0 + 1e-6:
                x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                x = np.clip(x, 0.0, 255.0).astype(np.uint8)
        else:
            x = x.astype(np.uint8)

    return x



def _discover_hl_url(hl_policy_dir: Optional[str] = None, port: int = DEFAULT_PORT) -> Optional[str]:
    here = pathlib.Path(__file__).resolve()
    # try some nearby folders for ip_eth0.txt
    for base in [here.parent, here.parent.parent, here.parents[2] if len(here.parents) > 2 else here.parent]:
        ip_file = base / IP_FILE_NAME
        if ip_file.exists():
            ip = ip_file.read_text().strip()
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
                return f"http://{ip}:{port}"
    return None

def _strip_code_fences(s: str) -> str:
    return re.sub(r"```[\s\S]*?```", "", s)

def _normalize_ws(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", s)

def _extract_ans_text(full_text: str) -> str:
    m = re.search(r"<\s*ans\s*>([\s\S]*?)<\s*/\s*ans\s*>", full_text, re.IGNORECASE)
    return m.group(1) if m else full_text

_POINT_ACTION_RE = re.compile(
    r"""
    \(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)            # (u, v)
    |
    <\s*action\s*>\s*(Open|Close)\s+Gripper\s*<\s*/\s*action\s*>   # action tag
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _parse_answer_block(ans_text: str) -> List[Waypoint2D]:
    waypoints: List[Waypoint2D] = []
    state_grip: Optional[int] = None
    last_was_point = False
    for tok in _POINT_ACTION_RE.finditer(ans_text):
        action = tok.group(3)
        if action:
            val = GRIPPER_OPEN if action.lower() == "open" else GRIPPER_CLOSE
            if last_was_point and waypoints and waypoints[-1].grip is None:
                waypoints[-1].grip = val
            state_grip = val
            last_was_point = False
            continue
        u = float(tok.group(1)); v = float(tok.group(2))
        u = max(0.0, min(1.0, u)); v = max(0.0, min(1.0, v))
        waypoints.append(Waypoint2D(u=u, v=v, grip=None))
        last_was_point = True
    fill = GRIPPER_OPEN if state_grip is None else state_grip
    for wp in waypoints:
        if wp.grip is None:
            wp.grip = fill
    return waypoints

def _encode_image_to_data_url(rgb: np.ndarray, *, resize_to: Optional[Tuple[int,int]] = None, jpeg_quality=95) -> Tuple[str, Tuple[int,int], np.ndarray]:
    rgb = _to_hwc_uint8(rgb)  # <--- NEW
    img = PILImage.fromarray(rgb)
    if resize_to is not None:
        img = img.resize(resize_to, resample=PILImage.BILINEAR)
    W, H = img.size
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", (W, H), np.array(img)


def _decode_b64_image_to_numpy(b64_or_data_url: str) -> Optional[np.ndarray]:
    if not b64_or_data_url:
        return None
    if b64_or_data_url.startswith("data:image"):
        _, b64 = b64_or_data_url.split(",", 1)
    else:
        b64 = b64_or_data_url
    try:
        raw = base64.b64decode(b64, validate=False)
        return np.array(PILImage.open(BytesIO(raw)).convert("RGB"))
    except Exception:
        return None

def _safe_get_content_text(result: Dict[str, Any]) -> Optional[str]:
    try:
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content and "text" in content[0]:
            return content[0]["text"]
    except Exception:
        pass
    for key in ("text", "generated_text"):
        if key in result and isinstance(result[key], str):
            return result[key]
    if "data" in result and isinstance(result["data"], list) and result["data"]:
        if isinstance(result["data"][0], str):
            return result["data"][0]
    return None

def _maybe_get_annotated_b64(result: Dict[str, Any]) -> Optional[str]:
    try:
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, list) and len(content) > 1:
            url = content[1].get("image_url", {}).get("url")
            if isinstance(url, str) and url.startswith("data:image"):
                return url
    except Exception:
        pass
    for key in ("image_b64", "annotated_image", "image"):
        v = result.get(key)
        if isinstance(v, str):
            return v
    if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 1:
        if isinstance(result["data"][1], str):
            return result["data"][1]
    return None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_snapshot_path(root: str, prefix: str, ext: str = "png") -> str:
    ensure_dir(root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.{ext}"
    return os.path.join(root, fname)

def save_image(rgb: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    PILImage.fromarray(rgb.astype(np.uint8)).save(path)

def _maybe_get_annotated_url(result: Dict[str, Any]) -> Optional[str]:
    """
    Try to find an HTTP/relative URL for the annotated image in common places.
    """
    # Gradio often returns {"data": [text, url_or_dict]}
    if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 1:
        v = result["data"][1]
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://") or v.startswith("/")):
            return v
        if isinstance(v, dict):
            # Some gradio backends return {"name": ".../file.png"} or {"url": "..."}
            for k in ("url", "name", "path"):
                if k in v and isinstance(v[k], str):
                    return v[k]
    # OpenAI-like content could include an image_url that is NOT a data URL
    try:
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, list) and len(content) > 1:
            url = content[1].get("image_url", {}).get("url")
            if isinstance(url, str) and (url.startswith("http") or url.startswith("/")):
                return url
    except Exception:
        pass
    return None

def _fetch_image_url_to_numpy(url: str, session: requests.Session, base_url: str) -> Optional[np.ndarray]:
    """
    Fetch an image by URL (absolute or relative to base_url) and decode to RGB numpy.
    """
    if not url:
        return None
    if url.startswith("/"):
        url = f"{base_url.rstrip('/')}{url}"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        return np.array(PILImage.open(BytesIO(r.content)).convert("RGB"))
    except Exception:
        return None


# ============================== Client ==============================

class HamsterVLMHTTP:
    """
    Minimal client for the Hamster VLM server.
    Primary: POST {base}/chat/completions (OpenAI-style)
    Fallback: POST {base}/predict or {base}/api/predict (Gradio-style)
    """
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = None):
        base_url = base_url or os.getenv("HL_VLM_URL") or _discover_hl_url(os.getenv("HL_POLICY_DIR"))
        if not base_url:
            base_url = f"http://127.0.0.1:{DEFAULT_PORT}"
            print(f"[HamsterVLMHTTP] Falling back to {base_url} (set HL_VLM_URL or ip_eth0.txt).")
        self.base_url = base_url.rstrip("/")
        self.model = os.getenv("HL_VLM_MODEL") or model or DEFAULT_MODEL
        self.timeout = float(os.getenv("HL_VLM_TIMEOUT", str(timeout if timeout is not None else 30.0)))
        self._session = requests.Session()

    def ping(self) -> bool:
        try:
            r = self._session.get(self.base_url, timeout=2.0)
            return r.ok
        except Exception:
            return False

    def get_path(
        self,
        rgb: np.ndarray,
        instruction: str,
        *,
        resize_to: Optional[Tuple[int,int]] = (512, 512),
        save_to: Optional[str] = None,
        return_image: bool = True,
        **extra_payload,
    ) -> VLMSketch:
        """
        Send an image + instruction, optionally resize to match the server/VLM,
        and (optionally) save the exact image sent under `save_to`.
        """
        data_url, (W,H), rgb_used = _encode_image_to_data_url(rgb, resize_to=resize_to)

        if save_to is not None:
            save_image(rgb_used, save_to)

        # Try OpenAI-like endpoint
        payload_openai = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text":
                        f"<quest>{instruction}</quest>\n"
                        "Return a single <ans>[(u,v), ...]</ans> list with (u,v)∈[0,1]. "
                        "You may include <action>Open/Close Gripper</action> tokens."
                    },
                ],
            }],
            **extra_payload,
        }

        t0 = time.time()
        try:
            result = self._post_json("/chat/completions", payload_openai)
        except Exception as e_openai:
            # Fallback to Gradio predict
            result = self._post_gradio_predict(data_url, instruction, extra_payload)

        raw_text = _safe_get_content_text(result) or ""
        print("\n--- RAW VLM TEXT START ---")
        print(raw_text)
        print("--- RAW VLM TEXT END ---\n")

        clean = _normalize_ws(_strip_code_fences(raw_text))
        ans_text = _extract_ans_text(clean)
        print("--- EXTRACTED <ans> BLOCK START ---")
        print(ans_text if ans_text else "(no <ans> block found; will parse full text)")
        print("--- EXTRACTED <ans> BLOCK END ---\n")

        waypoints = _parse_answer_block(ans_text)
        meta: Dict[str, Any] = {
            "latency_sec": time.time() - t0,
            "raw_text": raw_text,
            "ans_text": ans_text,
            "vlm_image_size": (W, H),
            "snapshot_path": save_to,
        }

        if return_image:
            # Try data URL / base64 first
            b64 = _maybe_get_annotated_b64(result)
            anno = _decode_b64_image_to_numpy(b64) if b64 else None
            if anno is None:
                # Try HTTP/relative URL next
                url = _maybe_get_annotated_url(result)
                if url:
                    anno = _fetch_image_url_to_numpy(url, self._session, self.base_url)
                    if anno is not None:
                        meta["annotated_image_url"] = url
            if anno is not None:
                meta["annotated_image"] = anno
                meta["annotated_image_shape"] = anno.shape
            else:
                # Stash raw hints for debugging
                if b64:
                    meta["annotated_image_b64_head"] = b64[:64]
                url = _maybe_get_annotated_url(result)
                if url:
                    meta["annotated_image_url"] = url

        return VLMSketch(waypoints=waypoints, meta=meta)

    # ---- internals ----
    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post_gradio_predict(self, data_url: str, instruction: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"data": [data_url, instruction]}
        if extra:
            payload["data"].append(extra)
        # try /predict then /api/predict
        for path in ("/predict", "/api/predict"):
            try:
                return self._post_json(path, payload)
            except Exception:
                continue
        raise RuntimeError("Gradio predict endpoints not available (/predict, /api/predict).")

# ============================== Camera snapshot helpers ==============================

def _get_rich_obs(env) -> Dict[str, Any]:
    """Attempt to fetch ManiSkill rich obs dict from an env (unwrapped)."""
    for name in ("get_obs", "get_observation", "_get_obs", "_get_obs_dict"):
        fn = getattr(env, name, None)
        if callable(fn):
            return fn()
    raise RuntimeError("Could not find a rich obs getter on the provided env.")

def get_camera_rgb_from_env(env, cam_name: str = "base_camera_1") -> np.ndarray:
    obs = _get_rich_obs(env.unwrapped if hasattr(env, "unwrapped") else env)
    rgb = obs["sensor_data"][cam_name]["rgb"]
    rgb = _to_hwc_uint8(rgb)  # <--- NEW
    return rgb


def save_annotated_image(sketch: VLMSketch, out_path: str, client: Optional["HamsterVLMHTTP"]=None) -> bool:
    """
    Save the annotated image to `out_path` if present in sketch.meta.
    Tries (1) np array, (2) data URL/base64, (3) HTTP/relative URL (uses client session if given).
    Returns True if saved, False otherwise.
    """
    ensure_dir(os.path.dirname(out_path))
    meta = sketch.meta or {}

    # 1) Direct ndarray
    anno = meta.get("annotated_image")
    if isinstance(anno, np.ndarray):
        save_image(anno, out_path)
        return True

    # 2) Base64 / data URL
    b64 = meta.get("annotated_image_b64") or meta.get("image_b64")
    if isinstance(b64, str) and b64:
        arr = _decode_b64_image_to_numpy(b64)
        if arr is not None:
            save_image(arr, out_path)
            return True

    # 3) URL
    url = meta.get("annotated_image_url")
    if isinstance(url, str) and url:
        session = client._session if client is not None else requests.Session()
        base = client.base_url if client is not None else "http://127.0.0.1"
        arr = _fetch_image_url_to_numpy(url, session, base)
        if arr is not None:
            save_image(arr, out_path)
            return True

    # Nothing found
    return False

# ===== Local annotation (client-side), mirroring the Gradio demo =====
def _to_pixel_points(waypoints, W, H):
    px = []
    grips = []
    for wp in waypoints:
        u, v = float(wp.u), float(wp.v)
        x = int(round(u * (W - 1)))
        y = int(round(v * (H - 1)))
        g = 1 if (getattr(wp, "grip", None) in (1, True)) else 0  # default close=0/open=1 like demo
        px.append((x, y))
        grips.append(g)
    return px, grips

def _draw_path_cv(rgb: np.ndarray, pixel_points, gripper_status, quest: str = None) -> np.ndarray:
    """Draw colored path + optional action markers using OpenCV, similar to the Gradio example."""
    import cv2
    from matplotlib import cm
    img = rgb.copy()
    H, W = img.shape[:2]

    # Scales relative to 512
    scale = max(min(W, H) / 512.0, 1.0)
    circle_radius = int(7 * scale)
    circle_thickness = max(1, int(2 * scale))
    line_thickness = max(1, int(2 * scale))
    font_scale = 0.5 * scale
    font_thickness = max(1, int(1 * scale))

    # Optional action circles
    for i, (x, y) in enumerate(pixel_points):
        if i == 0 or gripper_status[i] != gripper_status[i - 1]:
            # red for CLOSE (0), blue for OPEN (1) to match demo
            circle_color = (0, 0, 255) if gripper_status[i] == 0 else (255, 0, 0)
            cv2.circle(img, (x, y), circle_radius, circle_color, circle_thickness)

    # Interpolate along the path for smooth polyline
    if len(pixel_points) >= 2:
        pts = np.array(pixel_points, dtype=np.float32)
        # cumulative distances
        d = [0.0]
        for i in range(1, len(pts)):
            d.append(d[-1] + float(np.linalg.norm(pts[i] - pts[i-1])))
        total = d[-1] if d[-1] > 0 else 1.0
        samples = np.linspace(0, total, num=100)
        interp = []
        j = 0
        for s in samples:
            while j < len(d) - 2 and s > d[j + 1]:
                j += 1
            t = 0.0 if d[j+1] == d[j] else (s - d[j]) / (d[j+1] - d[j])
            p = (1 - t) * pts[j] + t * pts[j + 1]
            interp.append(p.astype(np.int32))
        interp = np.array(interp, dtype=np.int32)

        # jet colormap along the path
        cmap = cm.get_cmap('jet')
        colors = (cmap(np.linspace(0, 1, len(interp)))[:, :3] * 255).astype(np.uint8)

        for k in range(len(interp) - 1):
            pt1 = tuple(int(v) for v in interp[k])
            pt2 = tuple(int(v) for v in interp[k + 1])
            color = tuple(int(c) for c in colors[k])
            cv2.line(img, pt1, pt2, color=color, thickness=line_thickness)

    # Optional quest text overlay (top-left)
    if quest:
        cv2.rectangle(img, (5, 5), (5 + 320, 35), (0, 0, 0), -1)
        cv2.putText(img, quest[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return img

def _draw_path_pil(rgb: np.ndarray, pixel_points, gripper_status, quest: str = None) -> np.ndarray:
    """Fallback drawing with PIL if OpenCV isn't available (single-color line)."""
    from PIL import ImageDraw, ImageFont
    img = PILImage.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    # single color polyline
    if len(pixel_points) >= 2:
        draw.line(pixel_points, fill=(0, 255, 0), width=3)
    # action dots
    r = 5
    for i, (x, y) in enumerate(pixel_points):
        if i == 0 or gripper_status[i] != gripper_status[i - 1]:
            color = (255, 0, 0) if gripper_status[i] == 0 else (0, 0, 255)
            draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=2)
    # quest text
    if quest:
        draw.rectangle((5, 5, 5 + 320, 35), fill=(0, 0, 0))
        draw.text((10, 10), quest[:80], fill=(255, 255, 255))
    return np.array(img)

def save_local_annotated_from_sketch(snapshot_path: str, sketch, out_path: str, quest: str = None) -> bool:
    """
    Load the snapshot we saved (the exact image sent to the VLM), draw the VLM path locally,
    and save to out_path. Returns True on success.
    """
    if not os.path.exists(snapshot_path):
        return False
    rgb = np.array(PILImage.open(snapshot_path).convert("RGB"))
    H, W = rgb.shape[:2]
    px, grips = _to_pixel_points(sketch.waypoints, W, H)
    if len(px) == 0:
        return False
    try:
        # Prefer OpenCV path with nice gradients
        import cv2  # noqa: F401
        drawn = _draw_path_cv(rgb, px, grips, quest=quest)
    except Exception:
        drawn = _draw_path_pil(rgb, px, grips, quest=quest)
    save_image(drawn, out_path)
    return True

