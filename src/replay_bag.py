#!/usr/bin/env python3
"""Replay a ROS2 bag episode in MuJoCo (cam_base + cam_wrist sim vs real side-by-side).

Reads /current_pose TCP trajectory from the bag, solves IK per frame,
and renders a side-by-side video with real camera images for both cameras.

Usage:
    python replay_bag.py --bag-dir /path/to/episode_dir
    python replay_bag.py --bag-dir /path/to/episode_dir --fps 30 --output my_video.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, load_calib_wrist, make_model, get_model_ids, solve_ik,
    interpolate_pose, BagReader, HOME_QPOS, CAM_W, CAM_H, DEFAULT_CALIB,
    WRIST_CAM_INTRINSICS,
)
setup_egl()

import cv2
import imageio
import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Replay ROS2 bag in MuJoCo")
    ap.add_argument("--bag-dir", required=True)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--output", default=None, help="Output video path")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    episode = os.path.basename(args.bag_dir.rstrip("/"))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "replay_bag")
    os.makedirs(out_dir, exist_ok=True)
    output_path = args.output or os.path.join(out_dir, f"replay_{episode}.mp4")

    # ── Load bag ──
    print(f"\n=== Loading bag: {episode} ===")
    bag = BagReader(args.bag_dir)

    print("Reading /current_pose...")
    tcp_times, tcp_pos, tcp_quat = bag.read_pose_stamped("/current_pose")
    print(f"  {len(tcp_times)} msgs, duration: {tcp_times[-1]-tcp_times[0]:.2f}s")

    gripper_times, gripper_pos = None, None
    if "/franka_gripper/joint_states" in bag.topics:
        print("Reading /franka_gripper/joint_states...")
        gt, gp, _, _ = bag.read_joint_states("/franka_gripper/joint_states")
        if len(gt):
            gripper_times, gripper_pos = gt, gp

    # ── Load real images (base + wrist) ──
    real_base_images = []
    cam_base_topic = "/cam_base/camera/color/image_raw/compressed"
    if cam_base_topic in bag.topics:
        print(f"Reading {cam_base_topic}...")
        real_base_images = bag.read_compressed_images(cam_base_topic)
        print(f"  {len(real_base_images)} base images")

    real_wrist_images = []
    cam_wrist_topic = "/cam_wrist/camera/color/image_raw/compressed"
    if cam_wrist_topic in bag.topics:
        print(f"Reading {cam_wrist_topic}...")
        real_wrist_images = bag.read_compressed_images(cam_wrist_topic)
        print(f"  {len(real_wrist_images)} wrist images")
    else:
        print("  [WARN] No wrist camera topic found in bag")

    # ── Setup MuJoCo ──
    print("\n=== Setting up MuJoCo ===")
    T = load_calib(args.calib)
    load_calib_wrist(args.calib)
    model = make_model(T)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)

    cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    print(f"  cam_base id={cam_base_id}, cam_wrist id={cam_wrist_id}")

    wrist_w = WRIST_CAM_INTRINSICS["width"]
    wrist_h = WRIST_CAM_INTRINSICS["height"]

    renderer_base = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    renderer_wrist = mujoco.Renderer(model, height=wrist_h, width=wrist_w)

    frame_times = np.array([t for t, _ in real_base_images]) if real_base_images else tcp_times[::max(1, len(tcp_times)//300)]
    t0 = tcp_times[0]

    # Build wrist image lookup for nearest-timestamp matching
    wrist_ts = np.array([t for t, _ in real_wrist_images]) if real_wrist_images else np.array([])

    # ── Initial IK ──
    q_ref = HOME_QPOS.copy()
    for i, jid in enumerate(ids["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = q_ref[i]
    mujoco.mj_forward(model, data)
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             tcp_pos[0], tcp_quat[0], max_iter=500, q_ref=q_ref, ns_gain=1.0)

    print(f"\n=== Rendering {len(frame_times)} frames (base + wrist) ===")
    out_w, out_h = CAM_W // 2, CAM_H // 2
    frames_out = []

    for fi, t_frame in enumerate(frame_times):
        pos, quat = interpolate_pose(tcp_times, tcp_pos, tcp_quat, t_frame)
        pos_err, _ = solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
                              pos, quat, max_iter=50, q_ref=q_ref)

        if gripper_times is not None and len(gripper_times):
            g = np.interp(t_frame, gripper_times, gripper_pos[:, 0])
            for fid in ids["finger_ids"]:
                if fid >= 0:
                    data.qpos[model.jnt_qposadr[fid]] = g
            mujoco.mj_forward(model, data)

        # -- Render base cam --
        renderer_base.update_scene(data, camera=cam_base_id)
        sim_base = cv2.cvtColor(renderer_base.render().copy(), cv2.COLOR_RGB2BGR)
        sim_base = cv2.resize(sim_base, (out_w, out_h))

        # -- Render wrist cam --
        renderer_wrist.update_scene(data, camera=cam_wrist_id)
        sim_wrist = cv2.cvtColor(renderer_wrist.render().copy(), cv2.COLOR_RGB2BGR)
        sim_wrist = cv2.resize(sim_wrist, (out_w, out_h))

        # -- Real base image --
        if real_base_images and fi < len(real_base_images):
            real_base = cv2.resize(real_base_images[fi][1], (out_w, out_h))
        else:
            real_base = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # -- Real wrist image (nearest timestamp) --
        if len(wrist_ts) > 0:
            widx = np.argmin(np.abs(wrist_ts - t_frame))
            real_wrist = cv2.resize(real_wrist_images[widx][1], (out_w, out_h))
        else:
            real_wrist = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # -- Labels --
        cv2.putText(real_base, "REAL base", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(sim_base, "SIM base", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(real_wrist, "REAL wrist", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(sim_wrist, "SIM wrist", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # -- 2x2 grid: [real_base | sim_base] / [real_wrist | sim_wrist] --
        top_row = np.hstack([real_base, sim_base])
        bot_row = np.hstack([real_wrist, sim_wrist])
        frame = np.vstack([top_row, bot_row])
        frames_out.append(frame)

        if (fi+1) % 50 == 0 or fi == len(frame_times)-1:
            print(f"  {fi+1}/{len(frame_times)}  t={t_frame-t0:.2f}s  err={pos_err*1000:.1f}mm")

    renderer_base.close()
    renderer_wrist.close()

    # ── Write video ──
    print(f"\n=== Writing: {output_path} ===")
    writer = imageio.get_writer(output_path, fps=args.fps, codec='libx264',
                                 quality=8, pixelformat='yuv420p')
    for frame in frames_out:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()

    # Save comparison image (first frame)
    if frames_out:
        cv2.imwrite(os.path.join(out_dir, f"replay_{episode}_cmp.png"), frames_out[0])

    print(f"Done! {output_path}")


if __name__ == "__main__":
    main()
