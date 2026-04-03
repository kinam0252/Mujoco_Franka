#!/usr/bin/env python3
"""Replay a ROS2 bag episode in MuJoCo (cam_base sim vs real side-by-side).

Reads /current_pose TCP trajectory from the bag, solves IK per frame,
and renders a side-by-side video with real camera images.

Usage:
    python replay_bag.py --bag-dir /path/to/episode_dir
    python replay_bag.py --bag-dir /path/to/episode_dir --fps 30 --output my_video.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, make_model, get_model_ids, solve_ik,
    interpolate_pose, BagReader, HOME_QPOS, CAM_W, CAM_H, DEFAULT_CALIB,
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
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
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

    real_images = []
    cam_topic = "/cam_base/camera/color/image_raw/compressed"
    if cam_topic in bag.topics:
        print(f"Reading {cam_topic}...")
        real_images = bag.read_compressed_images(cam_topic)
        print(f"  {len(real_images)} images")

    # ── Setup MuJoCo ──
    print("\n=== Setting up MuJoCo ===")
    T = load_calib(args.calib)
    model = make_model(T)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

    frame_times = np.array([t for t, _ in real_images]) if real_images else tcp_times[::max(1, len(tcp_times)//300)]
    t0 = tcp_times[0]

    # ── Initial IK ──
    q_ref = HOME_QPOS.copy()
    for i, jid in enumerate(ids["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = q_ref[i]
    mujoco.mj_forward(model, data)
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             tcp_pos[0], tcp_quat[0], max_iter=500, q_ref=q_ref, ns_gain=1.0)

    print(f"\n=== Rendering {len(frame_times)} frames ===")
    out_w, out_h = CAM_W // 2, CAM_H // 2
    frames_sim, frames_real = [], []

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

        renderer.update_scene(data, camera=cam_id)
        sim = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
        frames_sim.append(cv2.resize(sim, (out_w, out_h)))
        if real_images and fi < len(real_images):
            frames_real.append(cv2.resize(real_images[fi][1], (out_w, out_h)))

        if (fi+1) % 50 == 0 or fi == len(frame_times)-1:
            print(f"  {fi+1}/{len(frame_times)}  t={t_frame-t0:.2f}s  err={pos_err*1000:.1f}mm")

    renderer.close()

    # ── Write video ──
    print(f"\n=== Writing: {output_path} ===")
    writer = imageio.get_writer(output_path, fps=args.fps, codec='libx264',
                                 quality=8, pixelformat='yuv420p')
    for i, sim_f in enumerate(frames_sim):
        if frames_real and i < len(frames_real):
            real_f = frames_real[i].copy()
            cv2.putText(real_f, "REAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(sim_f, "SIM", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            frame = np.hstack([real_f, sim_f])
        else:
            cv2.putText(sim_f, "SIM", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            frame = sim_f
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()

    # Save comparison image
    if frames_sim and frames_real:
        comp = np.hstack([frames_real[0], frames_sim[0]])
        cv2.putText(comp, "REAL", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(comp, "SIM", (out_w+10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imwrite(os.path.join(out_dir, f"replay_{episode}_cmp.png"), comp)

    print(f"Done! {output_path}")


if __name__ == "__main__":
    main()
