#!/usr/bin/env python3
"""Render Franka FR3 in static poses from multiple camera views.

Quick sanity check: loads the robot, sets a few poses, renders from
cam_base / front / side / top cameras, and saves a composite image.

Usage:
    python render_poses.py
    python render_poses.py --calib /path/to/calib.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, make_model, get_model_ids, set_robot_pose,
    render_camera, HOME_QPOS, CAM_W, CAM_H, DEFAULT_CALIB,
)
setup_egl()

import cv2
import mujoco
import numpy as np


POSES = {
    "ready": ([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], 0.04),
    "reach": ([0.0, 0.2, 0.0, -1.5, 0.0, 1.8, 0.785], 0.04),
    "grasp": ([0.0, 0.2, 0.0, -1.5, 0.0, 1.8, 0.785], 0.005),
}
CAMERAS = ["cam_base", "front", "side", "top"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    T = load_calib(args.calib)
    model = make_model(T)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)

    print(f"Rendering {len(POSES)} poses × {len(CAMERAS)} cameras...")
    cell_w, cell_h = 640, 360
    all_rows = []

    for pose_name, (joints, gripper) in POSES.items():
        set_robot_pose(model, data, ids["jnt_ids"], ids["finger_ids"], joints, gripper)
        row_imgs = []
        for cam in CAMERAS:
            img = render_camera(model, data, cam, cell_w, cell_h)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img_bgr, f"{cam} - {pose_name}", (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            row_imgs.append(img_bgr)
            # Save individual
            cv2.imwrite(os.path.join(out_dir, f"{pose_name}_{cam}.png"), img_bgr)
        all_rows.append(np.hstack(row_imgs))

    composite = np.vstack(all_rows)
    path = os.path.join(out_dir, "poses_composite.png")
    cv2.imwrite(path, composite)
    print(f"Saved: {path} ({composite.shape[1]}×{composite.shape[0]})")

    # Depth from cam_base (ready pose)
    set_robot_pose(model, data, ids["jnt_ids"], ids["finger_ids"], *POSES["ready"])
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="cam_base")
    depth = renderer.render()
    renderer.close()
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    depth_path = os.path.join(out_dir, "ready_cam_base_depth.png")
    cv2.imwrite(depth_path, cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO))
    print(f"Saved: {depth_path}")

    print("Done!")


if __name__ == "__main__":
    main()
