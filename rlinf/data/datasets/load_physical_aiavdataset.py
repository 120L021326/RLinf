# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load data for Alpamayo inference.

This module preserves the original official behavior when `physical_ai_av`
+ Hugging Face access are available, and adds a local-NCore path for offline
servers.

Local NCore mode is activated when either:
- `ncore_manifest_path` is passed, or
- environment variable `ALPAMAYO_NCORE_MANIFEST_PATH` is set.

Expected local files for one sequence look like:
    <root>/data.json
    <root>/pai_<clip>.ncore4.zarr.itar
    <root>/pai_<clip>.ncore4-camera_cross_left_120fov.zarr.itar
    <root>/pai_<clip>.ncore4-camera_front_wide_120fov.zarr.itar
    <root>/pai_<clip>.ncore4-camera_cross_right_120fov.zarr.itar
    <root>/pai_<clip>.ncore4-camera_front_tele_30fov.zarr.itar

Notes:
- Pose loading is adapted to the schema observed by the user: ego motion lives
  in `poses/default/dynamic_poses/.zattrs` under key `('rig', 'world')` with
  fields `poses` and `timestamps_us`.
- Camera stores vary more across exports. The loader below probes common Zarr
  layouts and raises an error with discovered arrays if it cannot find the
  frames/timestamps automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
import hashlib
import json
import re
import os
import tarfile
import warnings

import numpy as np
import scipy.spatial.transform as spt
from scipy.spatial.transform import Slerp
import torch
from einops import rearrange

try:
    import physical_ai_av  # type: ignore
except Exception:
    physical_ai_av = None

try:
    import zarr  # type: ignore
    _ZARR_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    zarr = None
    _ZARR_IMPORT_ERROR = e

try:
    import av  # type: ignore
except Exception:
    av = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


DEFAULT_CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}


class _NCoreSchemaError(RuntimeError):
    pass


@dataclass
class _ArrayInfo:
    path: str
    array: Any
    shape: tuple[int, ...]
    dtype: Any


class _ArrayInterpolator:
    """Linear interpolation for translation + SLERP for orientation."""

    def __init__(self, timestamps_us: np.ndarray, xyz: np.ndarray, quat_xyzw: np.ndarray):
        ts = np.asarray(timestamps_us, dtype=np.int64)
        xyz = np.asarray(xyz, dtype=np.float64)
        quat = np.asarray(quat_xyzw, dtype=np.float64)

        if ts.ndim != 1:
            raise ValueError(f"timestamps_us must be 1-D, got {ts.shape}")
        if len(ts) != len(xyz) or len(ts) != len(quat):
            raise ValueError(
                f"Length mismatch: timestamps={len(ts)} xyz={len(xyz)} quat={len(quat)}"
            )
        if len(ts) < 2:
            raise ValueError("Need at least two pose samples for interpolation")

        order = np.argsort(ts)
        self.ts = ts[order]
        self.xyz = xyz[order]
        self.rot = spt.Rotation.from_quat(quat[order])
        self.slerp = Slerp(self.ts.astype(np.float64), self.rot)

    def __call__(self, query_timestamps_us: np.ndarray) -> SimpleNamespace:
        q = np.asarray(query_timestamps_us, dtype=np.int64)
        qf = q.astype(np.float64)
        if qf.min() < self.ts[0] or qf.max() > self.ts[-1]:
            raise ValueError(
                f"Requested timestamps [{qf.min()}, {qf.max()}] exceed pose range "
                f"[{self.ts[0]}, {self.ts[-1]}]"
            )

        xyz_interp = np.stack([
            np.interp(qf, self.ts.astype(np.float64), self.xyz[:, i]) for i in range(3)
        ], axis=-1)
        rot_interp = self.slerp(qf)
        return SimpleNamespace(
            pose=SimpleNamespace(
                translation=xyz_interp,
                rotation=rot_interp,
            )
        )


def _resolve_local_paths(
    ncore_manifest_path: str | os.PathLike[str] | None,
    ncore_root: str | os.PathLike[str] | None,
    extract_cache_dir: str | os.PathLike[str] | None,
) -> tuple[Path | None, Path | None, Path | None]:
    manifest_env = os.environ.get("ALPAMAYO_NCORE_MANIFEST_PATH")
    root_env = os.environ.get("ALPAMAYO_NCORE_ROOT")
    cache_env = os.environ.get("ALPAMAYO_NCORE_EXTRACT_CACHE_DIR")

    manifest = Path(ncore_manifest_path or manifest_env).expanduser() if (ncore_manifest_path or manifest_env) else None
    root = Path(ncore_root or root_env).expanduser() if (ncore_root or root_env) else None
    cache = Path(extract_cache_dir or cache_env).expanduser() if (extract_cache_dir or cache_env) else None

    if manifest and manifest.is_dir():
        manifest = manifest / "data.json"
    if manifest and not root:
        root = manifest.parent
    return manifest, root, cache


def _safe_extract_tar(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:*") as tf:
        try:
            tf.extractall(dst, filter="data")  # py3.12+
        except TypeError:  # pragma: no cover
            tf.extractall(dst)


def _extract_itar_to_cache(itar_path: Path, extract_cache_dir: Path | None) -> Path:
    if not itar_path.exists():
        raise FileNotFoundError(f"Missing NCore store: {itar_path}")
    base = extract_cache_dir or (Path("/tmp") / "alpamayo_ncore_extract")
    base.mkdir(parents=True, exist_ok=True)
    st = itar_path.stat()
    sig = hashlib.md5(f"{itar_path.resolve()}::{st.st_size}::{st.st_mtime_ns}".encode()).hexdigest()[:12]
    out = base / f"{itar_path.name}.{sig}"
    marker = out / ".extract_done"
    if marker.exists():
        return out
    if out.exists():
        # clean partial extraction
        for p in sorted(out.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    _safe_extract_tar(itar_path, out)
    marker.write_text("ok\n")
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _store_from_manifest(manifest: dict[str, Any], component_match: str | None = None, camera_name: str | None = None) -> str:
    stores = manifest.get("component_stores", [])
    for store in stores:
        if component_match is None and camera_name is None:
            return store["path"]
        comps = store.get("components", {})
        if component_match and component_match in comps:
            return store["path"]
        if camera_name and "cameras" in comps and camera_name in comps["cameras"]:
            return store["path"]
    raise KeyError(f"Could not find store for component_match={component_match!r}, camera_name={camera_name!r}")


def _find_main_store_path(manifest: dict[str, Any], root: Path) -> Path:
    # Main store is the one that contains poses/intrinsics/cuboids/masks.
    for store in manifest.get("component_stores", []):
        comps = store.get("components", {})
        if any(k in comps for k in ("poses", "intrinsics", "cuboids", "masks")):
            return root / store["path"]
    # Fallback to first non-camera store.
    for store in manifest.get("component_stores", []):
        if "cameras" not in store.get("components", {}):
            return root / store["path"]
    raise KeyError("Could not identify main NCore store from manifest")


def _find_camera_store_paths(manifest: dict[str, Any], root: Path, camera_names: Iterable[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name in camera_names:
        path = _store_from_manifest(manifest, camera_name=name)
        out[name] = root / path
    return out


def _load_dynamic_pose_from_attrs(extracted_main_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attrs_path = extracted_main_dir / "poses" / "default" / "dynamic_poses" / ".zattrs"
    if not attrs_path.exists():
        raise _NCoreSchemaError(f"Missing dynamic pose attrs: {attrs_path}")
    obj = _read_json(attrs_path)
    key = "('rig', 'world')"
    if key not in obj:
        raise _NCoreSchemaError(
            f"Could not find {key!r} in {attrs_path}. Available keys: {list(obj.keys())[:20]}"
        )
    rig = obj[key]
    if "poses" not in rig or "timestamps_us" not in rig:
        raise _NCoreSchemaError(
            f"Expected keys 'poses' and 'timestamps_us' under {key!r}; got {list(rig.keys())}"
        )

    pose_mats = np.asarray(rig["poses"], dtype=np.float32)
    timestamps = np.asarray(rig["timestamps_us"], dtype=np.int64)

    if pose_mats.ndim != 3 or pose_mats.shape[0] != len(timestamps):
        raise _NCoreSchemaError(
            f"Unexpected pose matrix shape {pose_mats.shape} vs timestamps {timestamps.shape}"
        )
    if pose_mats.shape[-2:] == (4, 4):
        rotm = pose_mats[:, :3, :3]
        xyz = pose_mats[:, :3, 3]
    elif pose_mats.shape[-2:] == (3, 4):
        rotm = pose_mats[:, :3, :3]
        xyz = pose_mats[:, :3, 3]
    else:
        raise _NCoreSchemaError(f"Unsupported pose matrix shape: {pose_mats.shape}")

    quat = spt.Rotation.from_matrix(rotm).as_quat()
    return timestamps, xyz, quat


def _open_zarr_group(extracted_dir: Path):
    if zarr is None:
        raise ImportError(
            "Reading local NCore Zarr stores requires `zarr`, but importing it failed: "
            f"{_ZARR_IMPORT_ERROR!r}"
        )
    warnings.filterwarnings("ignore", message="Object at .* is not recognized as a component of a Zarr hierarchy")
    return zarr.open_group(str(extracted_dir), mode="r")


def _iter_arrays(group, prefix: str = "") -> list[_ArrayInfo]:
    out: list[_ArrayInfo] = []
    for name, arr in group.arrays():
        out.append(_ArrayInfo(
            path=f"{prefix}{name}" if prefix else name,
            array=arr,
            shape=tuple(arr.shape),
            dtype=arr.dtype,
        ))
    for name, sub in group.groups():
        sub_prefix = f"{prefix}{name}/" if prefix else f"{name}/"
        out.extend(_iter_arrays(sub, sub_prefix))
    return out


def _score_timestamp_candidate(info: _ArrayInfo) -> int:
    base = info.path.split("/")[-1].lower()
    score = 0
    if info.shape and len(info.shape) == 1:
        score += 10
    if np.issubdtype(np.dtype(info.dtype), np.integer):
        score += 10
    if base in {"timestamps_us", "timestamp_us", "timestamps", "timestamp", "time_us"}:
        score += 30
    if "timestamp" in base or "time" in base:
        score += 10
    return score


def _score_image_candidate(info: _ArrayInfo) -> int:
    base = info.path.split("/")[-1].lower()
    score = 0
    if base in {"images", "frames", "rgb", "rgba", "pixels", "data"}:
        score += 20
    if len(info.shape) == 4:
        score += 20
        if info.shape[-1] in (3, 4):
            score += 20
        if info.shape[1] in (3, 4):
            score += 10
    if len(info.shape) in (1, 3):
        # encoded frames or video blob are still possible
        score += 5
    return score


def _decode_image_bytes(blob: bytes) -> np.ndarray:
    if Image is None:
        raise RuntimeError("PIL is required to decode image bytes. Install pillow.")
    with Image.open(BytesIO(blob)) as img:
        img = img.convert("RGB")
        return np.asarray(img)


def _decode_video_bytes(blob: bytes) -> list[np.ndarray]:
    if av is None:
        raise RuntimeError("PyAV is required to decode video bytes. Install av.")
    out: list[np.ndarray] = []
    with av.open(BytesIO(blob)) as container:
        for frame in container.decode(video=0):
            out.append(frame.to_ndarray(format="rgb24"))
    if not out:
        raise RuntimeError("Decoded zero frames from video bytes")
    return out



def _extract_frame_timestamp_from_path(path: str) -> int | None:
    m = re.search(r"/frames/(\d+)/(?:image|rgb|rgba|data)$", path)
    if m:
        return int(m.group(1))
    return None


def _decode_scalar_blob(blob: Any) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, str):
        return blob.encode()
    try:
        return bytes(blob)
    except Exception as e:
        raise TypeError(f"Cannot convert scalar blob of type {type(blob)} to bytes: {e}") from e


def _try_load_camera_frames_from_path_embedded_timestamps(
    arrays: list[_ArrayInfo], image_timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    entries: list[tuple[int, _ArrayInfo]] = []
    for info in arrays:
        ts = _extract_frame_timestamp_from_path(info.path)
        if ts is None:
            continue
        if tuple(info.shape) != ():
            continue
        entries.append((ts, info))

    if not entries:
        return None

    entries.sort(key=lambda x: x[0])
    source_ts = np.asarray([ts for ts, _ in entries], dtype=np.int64)
    idx = _nearest_indices(source_ts, image_timestamps)

    frames: list[np.ndarray] = []
    picked_ts: list[int] = []
    for i in idx:
        ts, info = entries[int(i)]
        blob = _decode_scalar_blob(info.array[()])
        frame = _decode_image_bytes(blob)
        frames.append(frame)
        picked_ts.append(ts)

    return np.stack(frames, axis=0).astype(np.uint8), np.asarray(picked_ts, dtype=np.int64)

def _nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(source_ts, target_ts)
    idx = np.clip(idx, 0, len(source_ts) - 1)
    left = np.clip(idx - 1, 0, len(source_ts) - 1)
    choose_left = np.abs(source_ts[left] - target_ts) <= np.abs(source_ts[idx] - target_ts)
    return np.where(choose_left, left, idx)


def _load_frames_from_array(arr, indices: np.ndarray) -> np.ndarray:
    sample = arr[indices]
    sample = np.asarray(sample)
    if sample.ndim == 4 and sample.shape[-1] in (3, 4):
        frames = sample[..., :3]
    elif sample.ndim == 4 and sample.shape[1] in (3, 4):
        frames = np.transpose(sample[:, :3, :, :], (0, 2, 3, 1))
    elif sample.ndim == 1 and sample.dtype.kind in {"S", "O", "U"}:
        frames = np.stack([_decode_image_bytes(bytes(x)) for x in sample], axis=0)
    elif sample.ndim == 3 and sample.dtype.kind in {"S", "O", "U"}:
        # Uncommon, but keep a helpful error.
        raise _NCoreSchemaError(f"Unsupported encoded image array shape: {sample.shape}")
    else:
        raise _NCoreSchemaError(
            f"Unsupported frame array layout shape={sample.shape} dtype={sample.dtype}"
        )
    return np.asarray(frames, dtype=np.uint8)


def _try_load_camera_frames(extracted_cam_dir: Path, image_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    group = _open_zarr_group(extracted_cam_dir)
    arrays = _iter_arrays(group)
    if not arrays:
        raise _NCoreSchemaError(f"No arrays found in camera store: {extracted_cam_dir}")

    # NCore camera stores commonly encode each frame as a scalar byte blob at a path like
    # cameras/<camera_name>/frames/<timestamp_us>/image . In that layout, the timestamp is
    # embedded in the path and there is no standalone timestamp array.
    path_embedded = _try_load_camera_frames_from_path_embedded_timestamps(arrays, image_timestamps)
    if path_embedded is not None:
        return path_embedded

    ts_candidates = sorted(arrays, key=_score_timestamp_candidate, reverse=True)
    frame_candidates = sorted(arrays, key=_score_image_candidate, reverse=True)

    ts_info: _ArrayInfo | None = None
    source_ts: np.ndarray | None = None
    ts_errors: list[str] = []
    for cand in ts_candidates[:32]:
        try:
            # Real timestamp arrays should be 1-D numeric arrays. Skip scalar blobs/metadata.
            if len(cand.shape) != 1:
                ts_errors.append(f"{cand.path}: skipped shape={cand.shape}")
                continue
            arr = np.asarray(cand.array[:])
            if arr.ndim != 1 or arr.size == 0:
                ts_errors.append(f"{cand.path}: invalid ndim/size after read -> shape={arr.shape}")
                continue
            if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
                ts_errors.append(f"{cand.path}: skipped non-numeric dtype={arr.dtype}")
                continue
            source_ts = np.asarray(arr, dtype=np.int64).reshape(-1)
            ts_info = cand
            break
        except Exception as e:
            ts_errors.append(f"{cand.path}: {type(e).__name__}: {e}")
            continue

    if source_ts is None or ts_info is None:
        debug_lines = [f"- {a.path}  shape={a.shape} dtype={a.dtype}" for a in arrays[:120]]
        raise _NCoreSchemaError(
            "Failed to identify a usable camera timestamp array in local NCore camera store.\n"
            f"Store: {extracted_cam_dir}\n"
            "Timestamp candidate diagnostics:\n- " + "\n- ".join(ts_errors[:40]) + "\n"
            "Available arrays:\n" + "\n".join(debug_lines)
        )

    idx = _nearest_indices(source_ts, image_timestamps)

    last_error: Exception | None = None
    for frame_info in frame_candidates[:16]:
        try:
            # Skip scalar blobs here; they are handled in the video fallback below.
            if tuple(frame_info.shape) == ():
                continue
            frames = _load_frames_from_array(frame_info.array, idx)
            if len(frames) != len(image_timestamps):
                raise _NCoreSchemaError(
                    f"Decoded {len(frames)} frames from {frame_info.path}, expected {len(image_timestamps)}"
                )
            return frames, source_ts[idx]
        except Exception as e:
            last_error = e
            continue

    # Fallback: scalar byte blob that may contain a whole video.
    for info in frame_candidates[:16]:
        try:
            if tuple(info.shape) == ():
                blob = _decode_scalar_blob(info.array[()])
                decoded = _decode_video_bytes(blob)
                decoded = np.asarray(decoded, dtype=np.uint8)
                if len(decoded) != len(source_ts):
                    raise _NCoreSchemaError(
                        f"Decoded {len(decoded)} video frames from {info.path}, but timestamps has {len(source_ts)} entries"
                    )
                return decoded[idx], source_ts[idx]
        except Exception as e:
            last_error = e
            continue

    debug_lines = [f"- {a.path}  shape={a.shape} dtype={a.dtype}" for a in arrays[:120]]
    raise _NCoreSchemaError(
        "Failed to identify camera timestamps/frames in local NCore camera store.\n"
        f"Store: {extracted_cam_dir}\n"
        f"Chosen timestamp array: {ts_info.path} shape={ts_info.shape} dtype={ts_info.dtype}\n"
        f"Last decode error: {last_error!r}\n"
        "Available arrays:\n" + "\n".join(debug_lines)
    )

def _load_local_ncore(
    clip_id: str,
    t0_us: int,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
    camera_names: list[str],
    num_frames: int,
    ncore_manifest_path: Path,
    ncore_root: Path,
    extract_cache_dir: Path | None,
) -> dict[str, Any]:
    manifest = _read_json(ncore_manifest_path)

    main_store_path = _find_main_store_path(manifest, ncore_root)
    cam_store_paths = _find_camera_store_paths(manifest, ncore_root, camera_names)

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    extracted_main = _extract_itar_to_cache(main_store_path, extract_cache_dir)
    pose_timestamps, pose_xyz, pose_quat = _load_dynamic_pose_from_attrs(extracted_main)
    egomotion = _ArrayInterpolator(pose_timestamps, pose_xyz, pose_quat)

    if history_timestamps[0] < pose_timestamps[0] or future_timestamps[-1] > pose_timestamps[-1]:
        raise ValueError(
            f"Requested trajectory range [{history_timestamps[0]}, {future_timestamps[-1]}] exceeds "
            f"available ego pose range [{pose_timestamps[0]}, {pose_timestamps[-1]}]"
        )

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    image_frames_list: list[torch.Tensor] = []
    camera_indices_list: list[int] = []
    timestamps_list: list[torch.Tensor] = []

    for cam_name in camera_names:
        extracted_cam = _extract_itar_to_cache(cam_store_paths[cam_name], extract_cache_dir)
        frames, frame_timestamps = _try_load_camera_frames(extracted_cam, image_timestamps)
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(CAMERA_NAME_TO_INDEX.get(cam_name, 0))
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


def _load_official_physical_ai_av(
    clip_id: str,
    t0_us: int,
    avdi: Any,
    maybe_stream: bool,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
    camera_features: list[Any] | None,
    num_frames: int,
) -> dict[str, Any]:
    if physical_ai_av is None:
        raise ImportError(
            "physical_ai_av is not available. Provide local NCore paths or install the official package."
        )
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    egomotion = avdi.get_clip_feature(
        clip_id,
        avdi.features.LABELS.EGOMOTION,
        maybe_stream=maybe_stream,
    )

    assert t0_us > num_history_steps * time_step * 1_000_000, (
        "t0_us must be greater than the history time range"
    )

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(
            clip_id,
            cam_feature,
            maybe_stream=maybe_stream,
        )
        frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        if isinstance(cam_feature, str):
            cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
            cam_name = cam_name.lower()
        else:
            raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
        cam_idx = CAMERA_NAME_TO_INDEX.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi: Any | None = None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list[Any] | None = None,
    num_frames: int = 4,
    *,
    ncore_manifest_path: str | os.PathLike[str] | None = None,
    ncore_root: str | os.PathLike[str] | None = None,
    extract_cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load data for Alpamayo model inference.

    When local NCore paths are provided (or environment variables are set), the
    function reads the local NCore sequence. Otherwise it falls back to the
    original official `physical_ai_av` interface.
    """
    manifest_path, root, cache = _resolve_local_paths(ncore_manifest_path, ncore_root, extract_cache_dir)

    if manifest_path is not None:
        if root is None:
            raise ValueError("ncore_root could not be resolved")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing NCore manifest JSON: {manifest_path}")
        if camera_features is None:
            camera_names = DEFAULT_CAMERA_NAMES
        else:
            # In local mode we expect camera feature names/paths or plain camera names.
            camera_names = []
            for feat in camera_features:
                if isinstance(feat, str):
                    camera_names.append(feat.split("/")[-1].lower())
                else:
                    raise ValueError(
                        "In local NCore mode, camera_features should be strings or omitted"
                    )
        return _load_local_ncore(
            clip_id=clip_id,
            t0_us=t0_us,
            num_history_steps=num_history_steps,
            num_future_steps=num_future_steps,
            time_step=time_step,
            camera_names=camera_names,
            num_frames=num_frames,
            ncore_manifest_path=manifest_path,
            ncore_root=root,
            extract_cache_dir=cache,
        )

    return _load_official_physical_ai_av(
        clip_id=clip_id,
        t0_us=t0_us,
        avdi=avdi,
        maybe_stream=maybe_stream,
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
        time_step=time_step,
        camera_features=camera_features,
        num_frames=num_frames,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Alpamayo sample from official physical_ai_av or local NCore")
    parser.add_argument("--clip-id", default=os.environ.get("ALPAMAYO_CLIP_ID", "100ae358-f548-49b8-af4d-c0afdbcfe9ed"))
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--ncore-manifest-path", default=os.environ.get("ALPAMAYO_NCORE_MANIFEST_PATH"))
    parser.add_argument("--ncore-root", default=os.environ.get("ALPAMAYO_NCORE_ROOT"))
    parser.add_argument("--extract-cache-dir", default=os.environ.get("ALPAMAYO_NCORE_EXTRACT_CACHE_DIR"))
    args = parser.parse_args()

    data = load_physical_aiavdataset(
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        ncore_manifest_path="/workspace/dataset/pai_100ae358-f548-49b8-af4d-c0afdbcfe9ed.json",
        ncore_root="/workspace/dataset",
        extract_cache_dir="/tmp/alpamayo_ncore_extract",
    )
    print("Loaded keys:", list(data.keys()))
    print("image_frames:", tuple(data["image_frames"].shape))
    print("ego_history_xyz:", tuple(data["ego_history_xyz"].shape))
    print("ego_future_xyz:", tuple(data["ego_future_xyz"].shape))
