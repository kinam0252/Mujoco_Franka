#!/usr/bin/env python3
"""Render Franka FR3 from CSV EEF pose trajectories.

Reads eef_pose_quat.csv (pos_x,y,z, qx,qy,qz,qw, gripper_width),
solves IK, and renders sim images. Supports single-frame debug and batch mode.

Usage:
    # Debug first frame
    python render_csv.py --csv-dir /path/to/episode --mode first_frame

    # Batch all episodes
    python render_csv.py --data-dir /path/to/lift_data --mode batch --skip 1
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, make_model, get_model_ids, solve_ik,
    set_robot_pose, render_camera, HOME_QPOS, JOINT_NAMES, FINGER_NAMES,
    CAM_W, CAM_H, TCP_OFFSET, DEFAULT_CALIB,
)
setup_egl()

import cv2
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def load_csv(csv_path):
    """Load eef_pose_quat.csv → (N, 8) array."""
    return pd.read_csv(csv_path, header=0).values.astype(np.float64)


def ik_from_csv_row(model, data, ids, row, max_iter=200):
    """Solve IK for one CSV row [px,py,pz, qx,qy,qz,qw, gripper]."""
    pos, quat, gw = row[:3], row[3:7], row[7]
    pos_err, _ = solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
                          pos, quat, max_iter=max_iter)
    finger_pos = float(np.clip(gw, 0, 1)) * 0.04
    for fid in ids["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = finger_pos
    mujoco.mj_forward(model, data)
    return pos_err


def first_frame(model, data, renderer, ids, csv_path, out_path, real_img=None):
    """Render just the first frame (for camera alignment debugging)."""
    traj = load_csv(csv_path)
    # Init at home
    for i, jid in enumerate(ids["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
    mujoco.mj_forward(model, data)

    err = ik_from_csv_row(model, data, ids, traj[0], max_iter=500)
    print(f"  IK err: {err*1000:.1f}mm")

    renderer.update_scene(data, camera="cam_base")
    sim = renderer.render().copy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(sim, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path}")

    if real_img and os.path.isfile(real_img):
        real = cv2.imread(real_img)
        sim_bgr = cv2.resize(cv2.cvtColor(sim, cv2.COLOR_RGB2BGR), (real.shape[1], real.shape[0]))
        side = np.hstack([real, sim_bgr])
        cv2.putText(side, "REAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
        cv2.putText(side, "SIM", (real.shape[1]+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.imwrite(out_path.replace(".png", "_vs_real.png"), side)


def batch_episode(model, data, renderer, ids, csv_path, out_dir, skip=1, img_size=256):
    """Render every skip-th frame of an episode."""
    traj = load_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    expected = len(range(0, len(traj), skip))
    existing = len([f for f in os.listdir(out_dir) if f.endswith("_sim.png")]) if os.path.isdir(out_dir) else 0
    if existing >= expected > 0:
        print(f"  Skip ({existing}/{expected}): {out_dir}")
        return

    for i, jid in enumerate(ids["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
    mujoco.mj_forward(model, data)

    for idx in tqdm(range(0, len(traj), skip), desc="  frames", leave=False):
        ik_from_csv_row(model, data, ids, traj[idx])
        renderer.update_scene(data, camera="cam_base")
        sim = renderer.render().copy()
        sim_bgr = cv2.resize(cv2.cvtColor(sim, cv2.COLOR_RGB2BGR), (img_size, img_size))
        cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_sim.png"), sim_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["first_frame", "batch"], default="first_frame")
    ap.add_argument("--csv-dir", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--out", default=None)
    ap.add_argument("--real-img", default=None)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    T = load_calib(args.calib)
    model = make_model(T)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    if args.mode == "first_frame":
        csv_dir = args.csv_dir
        csv_path = os.path.join(csv_dir, "eef_pose_quat.csv")
        out_path = args.out or os.path.join(out_dir, "first_frame.png")
        first_frame(model, data, renderer, ids, csv_path, out_path, args.real_img)

    elif args.mode == "batch":
        data_dir = args.data_dir
        ep_dirs = sorted(glob.glob(os.path.join(data_dir, "kinam_*")))
        print(f"Found {len(ep_dirs)} episodes")
        for ep in tqdm(ep_dirs, desc="Episodes"):
            csv = os.path.join(ep, "eef_pose_quat.csv")
            if not os.path.isfile(csv):
                continue
            batch_episode(model, data, renderer, ids, csv,
                         os.path.join(args.out or out_dir, os.path.basename(ep)),
                         args.skip, args.img_size)

    renderer.close()
    print("Done!")


if __name__ == "__main__":
    main()
