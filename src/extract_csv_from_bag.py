#!/usr/bin/env python3
"""Extract EEF pose + joint positions from ROS2 bags → CSV.

For each episode, produces:
  - eef_pose_quat.csv: pos_x,y,z, qx,qy,qz,qw, gripper_width,
                        j1,j2,j3,j4,j5,j6,j7, finger1,finger2
  - task.txt: task description

Joint positions are sorted by joint number (fr3_joint1..7).
All signals are resampled to the target FPS using /current_pose timestamps
as the reference clock.

Usage:
    python extract_csv_from_bag.py \
        --input-dir /path/to/raw/Lift_v2 \
        --output-dir /path/to/lift_data_v2 \
        --fps 15
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import BagReader

JOINT_ORDER = [f"fr3_joint{i}" for i in range(1, 8)]


def process_episode(bag_dir, output_dir, fps=15):
    """Extract one episode from rosbag → CSV."""
    name = os.path.basename(bag_dir.rstrip("/"))
    out_dir = os.path.join(output_dir, name)
    csv_path = os.path.join(out_dir, "eef_pose_quat.csv")

    if os.path.exists(csv_path):
        print(f"  SKIP (already exists): {csv_path}")
        return True

    try:
        bag = BagReader(bag_dir)
    except Exception as e:
        print(f"  ERROR reading bag: {e}")
        return False

    # ── Read /current_pose (EEF) ──
    tcp_ts, tcp_pos, tcp_quat = bag.read_pose_stamped("/current_pose")
    if len(tcp_ts) < 10:
        print(f"  SKIP: too few pose msgs ({len(tcp_ts)})")
        return False
    print(f"  /current_pose: {len(tcp_ts)} msgs, {tcp_ts[-1]-tcp_ts[0]:.2f}s")

    # ── Read /franka/joint_states ──
    jnt_ts, jnt_pos, _, jnt_names = bag.read_joint_states("/franka/joint_states")
    # Sort to canonical order
    sort_idx = [list(jnt_names).index(jn) for jn in JOINT_ORDER]
    jnt_pos = jnt_pos[:, sort_idx]
    print(f"  /franka/joint_states: {len(jnt_ts)} msgs")

    # ── Read gripper ──
    # /gripper/joint_states has normalized width (0~1)
    grip_ts, grip_pos = None, None
    if "/gripper/joint_states" in bag.topics:
        g_ts, g_pos, _, _ = bag.read_joint_states("/gripper/joint_states")
        if len(g_ts) > 0:
            grip_ts, grip_pos = g_ts, g_pos[:, 0]  # single value
            print(f"  /gripper/joint_states: {len(grip_ts)} msgs, "
                  f"range [{grip_pos.min():.4f}, {grip_pos.max():.4f}]")

    # /franka_gripper/joint_states has finger positions in meters
    fgr_ts, fgr_pos = None, None
    if "/franka_gripper/joint_states" in bag.topics:
        fg_ts, fg_pos, _, _ = bag.read_joint_states("/franka_gripper/joint_states")
        if len(fg_ts) > 0:
            fgr_ts, fgr_pos = fg_ts, fg_pos  # (N, 2) finger1, finger2
            print(f"  /franka_gripper/joint_states: {len(fgr_ts)} msgs")

    # ── Resample at target FPS using /current_pose timestamps ──
    t0, t1 = tcp_ts[0], tcp_ts[-1]
    n_frames = int((t1 - t0) * fps) + 1
    frame_ts = np.linspace(t0, t1, n_frames)
    print(f"  Resampling to {n_frames} frames at {fps}fps")

    rows = []
    for t in frame_ts:
        # EEF pose (nearest)
        idx_tcp = np.argmin(np.abs(tcp_ts - t))
        px, py, pz = tcp_pos[idx_tcp]
        qx, qy, qz, qw = tcp_quat[idx_tcp]

        # Gripper width (normalized)
        gw = 1.0
        if grip_ts is not None:
            idx_g = np.argmin(np.abs(grip_ts - t))
            gw = float(grip_pos[idx_g])

        # Joint positions (interpolated)
        j = np.zeros(7)
        if len(jnt_ts) > 0:
            for d in range(7):
                j[d] = np.interp(t, jnt_ts, jnt_pos[:, d])

        # Finger positions
        f1, f2 = 0.04, 0.04
        if fgr_ts is not None:
            idx_f = np.argmin(np.abs(fgr_ts - t))
            f1 = float(fgr_pos[idx_f, 0])
            f2 = float(fgr_pos[idx_f, 1])

        rows.append([px, py, pz, qx, qy, qz, qw, gw,
                      j[0], j[1], j[2], j[3], j[4], j[5], j[6],
                      f1, f2])

    columns = ["pos_x", "pos_y", "pos_z", "qx", "qy", "qz", "qw", "gripper_width",
               "j1", "j2", "j3", "j4", "j5", "j6", "j7",
               "finger1", "finger2"]
    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.8f")

    # Copy task.txt
    task_file = os.path.join(bag_dir, "task.txt")
    if os.path.exists(task_file):
        import shutil
        shutil.copy2(task_file, os.path.join(out_dir, "task.txt"))
    else:
        with open(os.path.join(out_dir, "task.txt"), "w") as f:
            f.write("lift\n")

    print(f"  ✓ {name}: {len(df)} frames → {csv_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Extract EEF + joint CSV from rosbags")
    ap.add_argument("--input-dir", required=True, help="Dir with episode subdirs (rosbags)")
    ap.add_argument("--output-dir", required=True, help="Output dir for CSV episodes")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--single", type=str, default=None, help="Process single episode")
    args = ap.parse_args()

    if args.single:
        ok = process_episode(args.single, args.output_dir, args.fps)
        print(f"Result: {'OK' if ok else 'FAILED'}")
        return

    episodes = sorted([
        os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("kinam")
    ])
    print(f"=== Extracting {len(episodes)} episodes → {args.output_dir} ===\n")

    success, fail = 0, 0
    for i, ep in enumerate(episodes):
        print(f"[{i+1}/{len(episodes)}] {os.path.basename(ep)}")
        if process_episode(ep, args.output_dir, args.fps):
            success += 1
        else:
            fail += 1

    print(f"\n=== Done: {success} success, {fail} failed ===")


if __name__ == "__main__":
    main()
