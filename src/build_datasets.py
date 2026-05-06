#!/usr/bin/env python3
"""Build LeRobot v3 AND GR00T v2 datasets from CSV + MuJoCo sim videos.

Input:
  - CSV episodes: assets/lift_data/{episode}/eef_pose_quat.csv
  - Sim videos:   output/replay_csv/{episode}_cam_base.mp4, _cam_wrist.mp4

Output (two datasets):
  1) LeRobot v3.0:  --lerobot-dir  (data/chunk-000/, videos/chunk-000/, meta/)
  2) GR00T v2.1:    --groot-dir    (data/chunk-000/, videos/chunk-000/, meta/ + modality.json)

Both datasets share:
  - observation.state: [x, y, z, qx, qy, qz, qw, gripper_width] = 8D
  - action:            [x, y, z, qx, qy, qz, qw, gripper_width] = 8D (next-frame target)
  - observation.images.cam_base:  sim video (640x360, 15fps)
  - observation.images.cam_wrist: sim video (640x360, 15fps)
  - First 20 CSV frames are skipped (noisy), matching video content.

Usage:
  python build_datasets.py \
    --csv-dir /path/to/lift_data \
    --video-dir /path/to/replay_csv \
    --lerobot-dir /path/to/Lift_sim_lerobot \
    --groot-dir /path/to/Lift_sim_groot \
    --task "lift the wooden block"
"""
import argparse
import json
import os
import shutil
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SKIP_FRAMES = 20
FPS = 15
STATE_DIM = 8
ACTION_DIM = 8
VIDEO_H, VIDEO_W = 360, 640
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]
ACTION_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]


def find_episodes(csv_dir, video_dir):
    """Find episodes with matching CSV + base + wrist videos."""
    eps = sorted(glob.glob(os.path.join(csv_dir, "kinam*2026*")))
    valid = []
    for ep in eps:
        name = os.path.basename(ep)
        csv_path = os.path.join(ep, "eef_pose_quat.csv")
        vid_base = os.path.join(video_dir, f"{name}_cam_base.mp4")
        vid_wrist = os.path.join(video_dir, f"{name}_cam_wrist.mp4")
        if os.path.isfile(csv_path) and os.path.isfile(vid_base) and os.path.isfile(vid_wrist):
            valid.append((name, csv_path, vid_base, vid_wrist))
    return valid


def load_episode(csv_path):
    """Load CSV → (state[T,8], action[T,8], timestamps[T]) after skipping first 20 frames."""
    df = pd.read_csv(csv_path)
    df = df.iloc[SKIP_FRAMES:].reset_index(drop=True)
    T = len(df)

    state = df[["pos_x", "pos_y", "pos_z", "qx", "qy", "qz", "qw", "gripper_width"]].values.astype(np.float32)

    # Action = next-frame state (shift by 1, repeat last)
    action = np.roll(state, -1, axis=0).copy()
    action[-1] = state[-1]

    timestamps = np.arange(T, dtype=np.float64) / FPS
    return state, action, timestamps, T


def get_video_frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def compute_stats(arrays):
    """Compute mean/std/min/max/quantiles for normalization."""
    X = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "mean": X.mean(axis=0).tolist(),
        "std": np.maximum(X.std(axis=0), 1e-8).tolist(),
        "min": X.min(axis=0).tolist(),
        "max": X.max(axis=0).tolist(),
        "q01": np.quantile(X, 0.01, axis=0).tolist(),
        "q99": np.quantile(X, 0.99, axis=0).tolist(),
    }


# =====================================================================
# LeRobot v3 builder
# =====================================================================

def build_lerobot_v3(episodes_data, output_dir, task):
    """Build LeRobot v3.0 dataset."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    videos_dir = output_dir / "videos" / "chunk-000"
    meta_dir = output_dir / "meta"

    for d in [data_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cam in ["observation.images.cam_base", "observation.images.cam_wrist"]:
        (videos_dir / cam).mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    all_states, all_actions = [], []
    global_index = 0
    total_frames = 0

    for ep_idx, (name, state, action, timestamps, T, vid_base, vid_wrist) in enumerate(episodes_data):
        # Write parquet
        records = {
            "observation.state": [state[i].tolist() for i in range(T)],
            "action": [action[i].tolist() for i in range(T)],
            "timestamp": timestamps.tolist(),
            "frame_index": list(range(T)),
            "episode_index": [ep_idx] * T,
            "index": list(range(global_index, global_index + T)),
            "task_index": [0] * T,
        }
        pq.write_table(pa.table(records), data_dir / f"episode_{ep_idx:06d}.parquet")

        # Copy videos
        shutil.copy2(vid_base, videos_dir / "observation.images.cam_base" / f"episode_{ep_idx:06d}.mp4")
        shutil.copy2(vid_wrist, videos_dir / "observation.images.cam_wrist" / f"episode_{ep_idx:06d}.mp4")

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": T})
        all_states.append(state)
        all_actions.append(action)
        global_index += T
        total_frames += T

    # Meta files
    stats = {
        "observation.state": compute_stats(all_states),
        "action": compute_stats(all_actions),
    }

    video_features = {
        "dtype": "video",
        "shape": [VIDEO_H, VIDEO_W, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.fps": FPS,
            "video.height": VIDEO_H,
            "video.width": VIDEO_W,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "fr3",
        "total_episodes": len(episodes_data),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes_data) * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes_data)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ACTION_NAMES,
            },
            "observation.images.cam_base": video_features.copy(),
            "observation.images.cam_wrist": video_features.copy(),
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  LeRobot v3: {len(episodes_data)} episodes, {total_frames} frames → {output_dir}")


# =====================================================================
# GR00T v2 builder
# =====================================================================

def build_groot_v2(episodes_data, output_dir, task):
    """Build GR00T-compatible LeRobot v2 dataset with modality.json."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    videos_dir = output_dir / "videos" / "chunk-000"
    meta_dir = output_dir / "meta"

    for d in [data_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cam in ["observation.images.cam_base", "observation.images.cam_wrist"]:
        (videos_dir / cam).mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    all_states, all_actions = [], []
    global_index = 0
    total_frames = 0

    for ep_idx, (name, state, action, timestamps, T, vid_base, vid_wrist) in enumerate(episodes_data):
        # Write parquet
        records = {
            "observation.state": [state[i].tolist() for i in range(T)],
            "action": [action[i].tolist() for i in range(T)],
            "timestamp": timestamps.tolist(),
            "frame_index": list(range(T)),
            "episode_index": [ep_idx] * T,
            "index": list(range(global_index, global_index + T)),
            "task_index": [0] * T,
        }
        pq.write_table(pa.table(records), data_dir / f"episode_{ep_idx:06d}.parquet")

        # Copy videos
        shutil.copy2(vid_base, videos_dir / "observation.images.cam_base" / f"episode_{ep_idx:06d}.mp4")
        shutil.copy2(vid_wrist, videos_dir / "observation.images.cam_wrist" / f"episode_{ep_idx:06d}.mp4")

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": T})
        all_states.append(state)
        all_actions.append(action)
        global_index += T
        total_frames += T

    # Stats
    stats = {
        "observation.state": compute_stats(all_states),
        "action": compute_stats(all_actions),
    }

    # modality.json — GR00T-specific
    modality = {
        "state": {
            "proprio.eef_pos": {"start": 0, "end": 3},
            "proprio.eef_quat": {"start": 3, "end": 7},
            "proprio.gripper_width": {"start": 7, "end": 8},
        },
        "action": {
            "action.eef_pos": {"start": 0, "end": 3},
            "action.eef_quat": {"start": 3, "end": 7},
            "action.gripper_width": {"start": 7, "end": 8},
        },
        "video": {
            "cam_base": {"original_key": "observation.images.cam_base"},
            "cam_wrist": {"original_key": "observation.images.cam_wrist"},
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index"
            }
        },
    }

    video_features = {
        "dtype": "video",
        "shape": [VIDEO_H, VIDEO_W, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": VIDEO_H,
            "video.width": VIDEO_W,
            "video.fps": FPS,
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "fr3",
        "total_episodes": len(episodes_data),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes_data) * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes_data)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ACTION_NAMES,
            },
            "observation.images.cam_base": video_features.copy(),
            "observation.images.cam_wrist": video_features.copy(),
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(meta_dir / "relative_stats.json", "w") as f:
        json.dump({"action": stats["action"]}, f, indent=2)

    print(f"  GR00T v2:   {len(episodes_data)} episodes, {total_frames} frames → {output_dir}")


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Build LeRobot v3 + GR00T v2 datasets from CSV + sim videos")
    ap.add_argument("--csv-dir", required=True, help="Directory with episode CSVs")
    ap.add_argument("--video-dir", required=True, help="Directory with sim videos")
    ap.add_argument("--lerobot-dir", required=True, help="Output dir for LeRobot v3 dataset")
    ap.add_argument("--groot-dir", required=True, help="Output dir for GR00T v2 dataset")
    ap.add_argument("--task", default="lift the wooden block", help="Task description")
    args = ap.parse_args()

    # Find and validate episodes
    print("=== Finding episodes ===")
    episodes = find_episodes(args.csv_dir, args.video_dir)
    print(f"Found {len(episodes)} episodes with CSV + cam_base + cam_wrist videos")
    if not episodes:
        raise RuntimeError("No valid episodes found")

    # Load all episode data
    print("\n=== Loading episode data ===")
    episodes_data = []
    for i, (name, csv_path, vid_base, vid_wrist) in enumerate(episodes):
        state, action, timestamps, T = load_episode(csv_path)

        # Verify video frame count
        vid_frames = get_video_frame_count(vid_base)
        if vid_frames != T:
            print(f"  WARN: {name} csv_frames={T} vid_frames={vid_frames}, truncating")
            T = min(T, vid_frames)
            state, action, timestamps = state[:T], action[:T], timestamps[:T]

        episodes_data.append((name, state, action, timestamps, T, vid_base, vid_wrist))

        if (i + 1) % 10 == 0 or i == 0 or i == len(episodes) - 1:
            print(f"  [{i+1}/{len(episodes)}] {name}: {T} frames")

    total = sum(d[4] for d in episodes_data)
    print(f"  Total: {len(episodes_data)} episodes, {total} frames")

    # Build both datasets
    print("\n=== Building LeRobot v3 dataset ===")
    build_lerobot_v3(episodes_data, args.lerobot_dir, args.task)

    print("\n=== Building GR00T v2 dataset ===")
    build_groot_v2(episodes_data, args.groot_dir, args.task)

    print("\nDone!")


if __name__ == "__main__":
    main()
