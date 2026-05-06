#!/usr/bin/env python3
"""GR00T closed-loop inference in MuJoCo.

Loads a trained GR00T policy checkpoint directly (no server) and runs
closed-loop control of a Franka FR3 robot in MuJoCo to lift a cube.

Usage:
    python infer_gr00t_mujoco.py \
        --checkpoint /path/to/checkpoint-30000 \
        --task "lift the cube" \
        --output output/infer_result.mp4
"""
import argparse
import os
import subprocess
import sys
import time

# Headless EGL must be set before any MuJoCo/OpenGL import
os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, load_calib_wrist, make_model_with_cube,
    get_model_ids, solve_ik, get_tcp_pose, HOME_QPOS, CAM_H, CAM_W,
    WRIST_CAM_INTRINSICS,
)
setup_egl()

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Isaac-GR00T imports
GROOT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "Isaac-GR00T")
sys.path.insert(0, GROOT_ROOT)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

# ── Rendering resolution (match training data: 640×360) ──
RENDER_W = 640
RENDER_H = 360

# ── Defaults ──
DEFAULT_CHECKPOINT = os.path.join(
    os.path.expanduser("~"), "DATA", "INTERN", "training",
    "gr00t_groot_v2", "checkpoint-30000",
)
DEFAULT_CUBE_POS = [0.45, -0.05, 0.02]   # (x, y, z) on table
DEFAULT_CUBE_YAW_DEG = 0.0
CUBE_HALF_SIZE = (0.06, 0.02, 0.02)      # 12×4×4 cm lying flat
GRIPPER_CLOSE_THRESHOLD = 0.5
FPS = 15


def parse_args():
    p = argparse.ArgumentParser(description="GR00T MuJoCo inference")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to GR00T checkpoint directory")
    p.add_argument("--task", default="lift the cube",
                   help="Language instruction for the policy")
    p.add_argument("--cube-pos", type=float, nargs=3, default=DEFAULT_CUBE_POS,
                   metavar=("X", "Y", "Z"),
                   help="Initial cube position (world frame)")
    p.add_argument("--cube-yaw", type=float, default=DEFAULT_CUBE_YAW_DEG,
                   help="Initial cube yaw angle (degrees)")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Maximum inference steps")
    p.add_argument("--open-loop-horizon", type=int, default=8,
                   help="Number of actions to execute per chunk before re-querying")
    p.add_argument("--ema-alpha", type=float, default=0.0,
                   help="EMA smoothing factor (0=off, 0.5=moderate, 0.8=heavy)")
    p.add_argument("--device", default="cuda:0",
                   help="Device for model inference")
    p.add_argument("--output", default=None,
                   help="Output video path (default: output/infer_<timestamp>.mp4)")
    p.add_argument("--calib", default=None,
                   help="Camera calibration YAML path")
    p.add_argument("--scene-xml", default=None,
                   help="Path to fr3_with_hand.xml (auto-detected if not set)")
    return p.parse_args()


# =====================================================================
# MuJoCo environment helpers
# =====================================================================

def init_scene(args):
    """Build MuJoCo scene with cube and return (model, data, ids, renderers)."""
    T_base_cam = load_calib(args.calib)
    load_calib_wrist(args.calib)

    cube_pos = np.array(args.cube_pos, dtype=np.float64)
    yaw = np.radians(args.cube_yaw)
    q_xyzw = Rotation.from_euler('z', yaw).as_quat()
    cube_quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    model = make_model_with_cube(T_base_cam, cube_pos, cube_quat_wxyz,
                                  cube_size=CUBE_HALF_SIZE,
                                  scene_xml=args.scene_xml)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)

    # Camera IDs
    cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")

    # Renderers at training-data resolution
    renderer_base = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    renderer_wrist = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    # Cube joint for kinematic manipulation
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qposadr = model.jnt_qposadr[cube_jnt_id] if cube_jnt_id >= 0 else None

    # Set cube initial position
    if cube_qposadr is not None:
        data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quat_wxyz

    # Reset robot to home pose
    for i, jid in enumerate(ids["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
    for fid in ids["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04  # fully open
    mujoco.mj_forward(model, data)

    return {
        "model": model, "data": data, "ids": ids,
        "renderer_base": renderer_base, "renderer_wrist": renderer_wrist,
        "cam_base_id": cam_base_id, "cam_wrist_id": cam_wrist_id,
        "cube_qposadr": cube_qposadr,
        "cube_init_pos": cube_pos.copy(),
        "cube_init_quat_wxyz": list(cube_quat_wxyz),
    }


def get_observation(env, task_str):
    """Capture current observation from MuJoCo and format for GR00T.

    Returns dict with video, state, language matching Gr00tPolicy input format.
    """
    model, data, ids = env["model"], env["data"], env["ids"]

    # ── Render cameras (RGB, uint8) ──
    env["renderer_base"].update_scene(data, camera=env["cam_base_id"])
    img_base = env["renderer_base"].render().copy()  # (H, W, 3) RGB

    env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"])
    img_wrist = env["renderer_wrist"].render().copy()

    # ── EEF state ──
    tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
    # MuJoCo xquat is wxyz; convert rotation matrix to xyzw for GR00T
    eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()  # xyzw

    # Gripper width: normalized 0~1 (finger joint range 0..0.04)
    finger_id = ids["finger_ids"][0]
    if finger_id >= 0:
        finger_pos = data.qpos[model.jnt_qposadr[finger_id]]
        gripper_width = np.clip(finger_pos / 0.04, 0.0, 1.0)
    else:
        gripper_width = 1.0

    # ── Format as GR00T observation ──
    observation = {
        "video": {
            "cam_base": img_base[np.newaxis, np.newaxis].astype(np.uint8),    # (1,1,H,W,3)
            "cam_wrist": img_wrist[np.newaxis, np.newaxis].astype(np.uint8),
        },
        "state": {
            "proprio.eef_pos": tcp_pos[np.newaxis, np.newaxis].astype(np.float32),       # (1,1,3)
            "proprio.eef_quat": eef_quat_xyzw[np.newaxis, np.newaxis].astype(np.float32), # (1,1,4)
            "proprio.gripper_width": np.array([[[gripper_width]]], dtype=np.float32),      # (1,1,1)
        },
        "language": {
            "annotation.human.action.task_description": [[task_str]],  # list[list[str]] (B=1, T=1)
        },
    }
    return observation


def apply_action(env, eef_pos, eef_quat_xyzw, gripper_width, grasp_state):
    """Apply a single action step to MuJoCo via IK.

    Args:
        env: environment dict
        eef_pos: (3,) target EEF position
        eef_quat_xyzw: (4,) target EEF quaternion (xyzw)
        gripper_width: float 0~1
        grasp_state: dict tracking cube grasp (modified in-place)
    """
    model, data, ids = env["model"], env["data"], env["ids"]

    # Normalize quaternion
    quat_norm = np.linalg.norm(eef_quat_xyzw)
    if quat_norm > 1e-6:
        eef_quat_xyzw = eef_quat_xyzw / quat_norm

    # IK solve
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             eef_pos, eef_quat_xyzw, max_iter=50)

    # Set gripper
    finger_pos = np.clip(gripper_width, 0.0, 1.0) * 0.04
    for fid in ids["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = finger_pos

    # ── Cube kinematic attachment ──
    cube_qposadr = env["cube_qposadr"]
    if cube_qposadr is not None:
        if not grasp_state["grasped"] and gripper_width < GRIPPER_CLOSE_THRESHOLD:
            # Gripper just closed → compute cube-in-TCP relative transform
            tcp_pos_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
            T_tcp = np.eye(4)
            T_tcp[:3, :3] = tcp_R_now
            T_tcp[:3, 3] = tcp_pos_now

            cube_pos_now = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            cube_quat_wxyz = data.qpos[cube_qposadr + 3:cube_qposadr + 7].copy()
            cube_quat_xyzw = [cube_quat_wxyz[1], cube_quat_wxyz[2],
                               cube_quat_wxyz[3], cube_quat_wxyz[0]]
            T_cube = np.eye(4)
            T_cube[:3, :3] = Rotation.from_quat(cube_quat_xyzw).as_matrix()
            T_cube[:3, 3] = cube_pos_now

            grasp_state["grasped"] = True
            grasp_state["T_cube_in_tcp"] = np.linalg.inv(T_tcp) @ T_cube

        if grasp_state["grasped"]:
            # Move cube with TCP
            tcp_pos_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
            T_tcp = np.eye(4)
            T_tcp[:3, :3] = tcp_R_now
            T_tcp[:3, 3] = tcp_pos_now
            T_cube = T_tcp @ grasp_state["T_cube_in_tcp"]

            data.qpos[cube_qposadr:cube_qposadr + 3] = T_cube[:3, 3]
            q_c = Rotation.from_matrix(T_cube[:3, :3]).as_quat()  # xyzw
            data.qpos[cube_qposadr + 3:cube_qposadr + 7] = [
                q_c[3], q_c[0], q_c[1], q_c[2]]  # wxyz

    mujoco.mj_forward(model, data)


def write_video(frames, output_path, fps):
    """Write BGR frames to mp4 via ffmpeg."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    h, w = frames[0].shape[:2]
    w = w & ~1; h = h & ~1
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp.mp4"

    proc = subprocess.Popen([
        ffmpeg_exe, '-y', '-loglevel', 'warning',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '23', '-preset', 'fast', tmp_path,
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        os.rename(tmp_path, output_path)
    else:
        stderr = proc.stderr.read()
        print(f"ffmpeg error: {stderr.decode()[:500]}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =====================================================================
# Main inference loop
# =====================================================================

def main():
    args = parse_args()

    # ── Output path ──
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "inference")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"infer_{ts}.mp4")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Task       : {args.task}")
    print(f"Cube pos   : {args.cube_pos}")
    print(f"Max steps  : {args.max_steps}")
    print(f"Horizon    : {args.open_loop_horizon}")
    print(f"Device     : {args.device}")
    print(f"Output     : {args.output}")

    # ── Load GR00T policy ──
    print("\nLoading GR00T policy...")
    t0 = time.time()
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint,
        device=args.device,
    )
    print(f"Policy loaded in {time.time() - t0:.1f}s")

    # ── Initialize MuJoCo scene ──
    print("Initializing MuJoCo scene...")
    env = init_scene(args)
    model, data, ids = env["model"], env["data"], env["ids"]

    # IK to a reasonable initial pose (above cube)
    init_pos = np.array([args.cube_pos[0], args.cube_pos[1], 0.25])
    init_quat = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()  # point down
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             init_pos, init_quat, max_iter=500, ns_gain=1.0)
    mujoco.mj_forward(model, data)
    print("Scene ready.\n")

    # ── Inference loop ──
    action_chunk = None
    chunk_idx = args.open_loop_horizon  # force first query
    grasp_state = {"grasped": False, "T_cube_in_tcp": None}
    frames_base = []
    frames_wrist = []

    # EMA smoothing state
    ema_alpha = args.ema_alpha
    ema_pos = None
    ema_quat = None

    print(f"Running inference... (EMA alpha={ema_alpha})")
    for step in range(args.max_steps):
        # ── Query policy if chunk exhausted ──
        if chunk_idx >= args.open_loop_horizon:
            t_infer = time.time()
            obs = get_observation(env, args.task)
            action_result, info = policy.get_action(obs)
            # action_result: {"action.eef_pos": (1,16,3), "action.eef_quat": (1,16,4), "action.gripper_width": (1,16,1)}
            action_chunk = {
                "eef_pos": action_result["action.eef_pos"][0],          # (16, 3)
                "eef_quat": action_result["action.eef_quat"][0],        # (16, 4)
                "gripper_width": action_result["action.gripper_width"][0],  # (16, 1)
            }
            chunk_idx = 0
            dt_infer = time.time() - t_infer
            print(f"  Step {step:>3d}: new chunk (inference {dt_infer:.3f}s)")

        # ── Extract and apply current action ──
        pos = action_chunk["eef_pos"][chunk_idx].copy()    # (3,)
        quat = action_chunk["eef_quat"][chunk_idx].copy()  # (4,) xyzw
        gw = float(action_chunk["gripper_width"][chunk_idx, 0])

        # ── EMA smoothing ──
        if ema_alpha > 0:
            if ema_pos is None:
                ema_pos = pos.copy()
                ema_quat = quat.copy()
            else:
                ema_pos = ema_alpha * ema_pos + (1 - ema_alpha) * pos
                ema_quat = ema_alpha * ema_quat + (1 - ema_alpha) * quat
                ema_quat = ema_quat / np.linalg.norm(ema_quat)  # re-normalize
            pos = ema_pos.copy()
            quat = ema_quat.copy()

        apply_action(env, pos, quat, gw, grasp_state)
        chunk_idx += 1

        # ── Record frames ──
        env["renderer_base"].update_scene(data, camera=env["cam_base_id"])
        frame_base = cv2.cvtColor(env["renderer_base"].render().copy(), cv2.COLOR_RGB2BGR)
        frames_base.append(frame_base)

        env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"])
        frame_wrist = cv2.cvtColor(env["renderer_wrist"].render().copy(), cv2.COLOR_RGB2BGR)
        frames_wrist.append(frame_wrist)

        # Print progress
        if (step + 1) % 50 == 0:
            tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
            print(f"  Step {step + 1:>3d}: eef=({tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f})"
                  f"  gw={gw:.3f}  grasped={grasp_state['grasped']}")

    # ── Evaluate success ──
    cube_qposadr = env["cube_qposadr"]
    if cube_qposadr is not None:
        cube_z = data.qpos[cube_qposadr + 2]
        init_z = args.cube_pos[2]
        lift_cm = max(0.0, (cube_z - init_z) * 100.0)
        success = lift_cm >= 4.0
        print(f"\nResult: lift={lift_cm:.1f}cm  {'SUCCESS' if success else 'FAIL'}")
    else:
        print("\nNo cube in scene — cannot evaluate lift.")

    # ── Save video (side-by-side cam_base + cam_wrist) ──
    print(f"Saving video to {args.output}...")
    combined = []
    for fb, fw in zip(frames_base, frames_wrist):
        combined.append(np.concatenate([fb, fw], axis=1))
    write_video(combined, args.output, FPS)
    print("Done.")

    # Cleanup
    env["renderer_base"].close()
    env["renderer_wrist"].close()


if __name__ == "__main__":
    main()
