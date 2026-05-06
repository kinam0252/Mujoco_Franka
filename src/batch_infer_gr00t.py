#!/usr/bin/env python3
"""Batch GR00T inference over all episodes.

For each episode in --data-dir, extracts cube position from the CSV
(gripper close frame), runs GR00T closed-loop inference in MuJoCo,
and saves a side-by-side video.

Usage:
    python batch_infer_gr00t.py --data-dir /path/to/lift_data
"""
import argparse
import glob
import os
import sys
import time

os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, os.path.dirname(__file__))
from utils import setup_egl
setup_egl()

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


GRIPPER_CLOSE_THRESHOLD = 0.7
SCENE_XML = os.path.join(
    os.path.dirname(__file__), "..", "..", "residual-offpolicy-rl_v1",
    "mujoco_menagerie", "franka_fr3", "fr3_with_hand.xml",
)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.expanduser("~"), "DATA", "INTERN", "training",
    "gr00t_groot_v2", "checkpoint-30000",
)


def extract_cube_pose(csv_path):
    """Extract cube position and yaw from episode CSV (gripper close frame)."""
    df = pd.read_csv(csv_path)
    gw = df["gripper_width"].values
    mask = gw < GRIPPER_CLOSE_THRESHOLD
    if not mask.any():
        return None, None

    close_idx = int(np.argmax(mask))
    row = df.iloc[close_idx]
    close_pos = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
    close_quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]])

    R_grip = Rotation.from_quat(close_quat).as_matrix()
    grip_x = R_grip[:, 0]
    grip_x_horiz = np.array([grip_x[0], grip_x[1], 0.0])
    n = np.linalg.norm(grip_x_horiz)
    if n > 1e-6:
        grip_x_horiz /= n
    else:
        grip_x_horiz = np.array([1.0, 0.0, 0.0])
    yaw_deg = np.degrees(np.arctan2(grip_x_horiz[1], grip_x_horiz[0]))

    cube_pos = [float(close_pos[0]), float(close_pos[1]), 0.02]
    return cube_pos, float(yaw_deg)


def parse_args():
    p = argparse.ArgumentParser(description="Batch GR00T MuJoCo inference")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing episode folders (each with eef_pose_quat.csv)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="GR00T checkpoint path")
    p.add_argument("--task", default="lift the cube")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--open-loop-horizon", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.0,
                   help="EMA smoothing factor (0=off, 0.9=recommended)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: Mujoco_Franka/output/batch_infer/)")
    p.add_argument("--scene-xml", default=None)
    p.add_argument("--calib", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    scene_xml = args.scene_xml or SCENE_XML
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "batch_infer")
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all episodes
    episodes = sorted(glob.glob(os.path.join(args.data_dir, "kinam_2026*")))
    print(f"Found {len(episodes)} episodes in {args.data_dir}")

    # Pre-extract cube poses before loading model (fast)
    episode_info = []
    for ep_dir in episodes:
        name = os.path.basename(ep_dir)
        csv_path = os.path.join(ep_dir, "eef_pose_quat.csv")
        if not os.path.isfile(csv_path):
            print(f"  SKIP {name}: no CSV")
            continue

        out_path = os.path.join(args.output_dir, f"{name}.mp4")
        if os.path.exists(out_path):
            print(f"  SKIP {name}: already done")
            continue

        cube_pos, cube_yaw = extract_cube_pose(csv_path)
        if cube_pos is None:
            print(f"  SKIP {name}: no gripper close")
            continue

        episode_info.append({
            "name": name,
            "cube_pos": cube_pos,
            "cube_yaw": cube_yaw,
            "out_path": out_path,
        })

    if not episode_info:
        print("No episodes to process.")
        return

    print(f"\n{len(episode_info)} episodes to process. Loading model...")

    # Import heavy modules and load model once
    from infer_gr00t_mujoco import (
        init_scene, get_observation, apply_action, write_video,
        Gr00tPolicy, EmbodimentTag, solve_ik, get_tcp_pose,
        mujoco, cv2, Rotation,
        RENDER_H, RENDER_W, FPS,
    )
    from utils import load_calib, load_calib_wrist, HOME_QPOS

    t0 = time.time()
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint,
        device=args.device,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    results = []

    for idx, ep in enumerate(episode_info):
        print(f"[{idx+1}/{len(episode_info)}] {ep['name']}  cube=({ep['cube_pos'][0]:.3f}, {ep['cube_pos'][1]:.3f})  yaw={ep['cube_yaw']:.1f}°")

        # Build args namespace for init_scene
        scene_args = argparse.Namespace(
            cube_pos=ep["cube_pos"],
            cube_yaw=ep["cube_yaw"],
            calib=args.calib,
            scene_xml=scene_xml,
        )

        env = init_scene(scene_args)
        model_mj, data, ids = env["model"], env["data"], env["ids"]

        # Init robot above cube
        init_pos = np.array([ep["cube_pos"][0], ep["cube_pos"][1], 0.25])
        init_quat = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        solve_ik(model_mj, data, ids["hand_id"], ids["jnt_ids"],
                 init_pos, init_quat, max_iter=500, ns_gain=1.0)
        mujoco.mj_forward(model_mj, data)

        # Inference loop
        action_chunk = None
        chunk_idx = args.open_loop_horizon
        grasp_state = {"grasped": False, "T_cube_in_tcp": None}
        frames_base = []
        frames_wrist = []
        ema_pos = None
        ema_quat = None

        for step in range(args.max_steps):
            if chunk_idx >= args.open_loop_horizon:
                obs = get_observation(env, args.task)
                action_result, _ = policy.get_action(obs)
                action_chunk = {
                    "eef_pos": action_result["action.eef_pos"][0],
                    "eef_quat": action_result["action.eef_quat"][0],
                    "gripper_width": action_result["action.gripper_width"][0],
                }
                chunk_idx = 0

            pos = action_chunk["eef_pos"][chunk_idx].copy()
            quat = action_chunk["eef_quat"][chunk_idx].copy()
            gw = float(action_chunk["gripper_width"][chunk_idx, 0])

            # EMA smoothing
            if args.ema_alpha > 0:
                if ema_pos is None:
                    ema_pos, ema_quat = pos.copy(), quat.copy()
                else:
                    ema_pos = args.ema_alpha * ema_pos + (1 - args.ema_alpha) * pos
                    ema_quat = args.ema_alpha * ema_quat + (1 - args.ema_alpha) * quat
                    ema_quat /= np.linalg.norm(ema_quat)
                pos, quat = ema_pos.copy(), ema_quat.copy()

            apply_action(env, pos, quat, gw, grasp_state)
            chunk_idx += 1

            env["renderer_base"].update_scene(data, camera=env["cam_base_id"])
            fb = cv2.cvtColor(env["renderer_base"].render().copy(), cv2.COLOR_RGB2BGR)
            frames_base.append(fb)

            env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"])
            fw = cv2.cvtColor(env["renderer_wrist"].render().copy(), cv2.COLOR_RGB2BGR)
            frames_wrist.append(fw)

        # Evaluate
        cube_qposadr = env["cube_qposadr"]
        lift_cm = 0.0
        success = False
        if cube_qposadr is not None:
            cube_z = data.qpos[cube_qposadr + 2]
            lift_cm = max(0.0, (cube_z - ep["cube_pos"][2]) * 100.0)
            success = lift_cm >= 4.0

        tag = "SUCCESS" if success else "FAIL"
        print(f"  → lift={lift_cm:.1f}cm  {tag}")
        results.append({"episode": ep["name"], "lift_cm": lift_cm, "success": success})

        # Save video
        combined = [np.concatenate([fb, fw], axis=1) for fb, fw in zip(frames_base, frames_wrist)]
        write_video(combined, ep["out_path"], FPS)

        # Cleanup renderers
        env["renderer_base"].close()
        env["renderer_wrist"].close()

    # Summary
    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*50}")
    print(f"Results: {n_success}/{len(results)} success ({100*n_success/len(results):.0f}%)")
    print(f"Mean lift: {np.mean([r['lift_cm'] for r in results]):.1f}cm")
    print(f"Videos saved to: {args.output_dir}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Results CSV: {csv_path}")


if __name__ == "__main__":
    main()
