#!/usr/bin/env python3
"""ACT closed-loop inference in MuJoCo.

Loads a trained ACT (Action Chunking with Transformers) policy checkpoint
and runs closed-loop control of a Franka FR3 robot in MuJoCo to lift a cube.

Usage:
    python infer_act_mujoco.py \
        --checkpoint /path/to/act_lerobot_224/checkpoints/030000/pretrained_model \
        --output output/act_infer.mp4

    # Compare both checkpoints:
    python infer_act_mujoco.py \
        --checkpoint /path/to/act_lerobot/checkpoints/030000/pretrained_model \
        --output output/act_640.mp4

    python infer_act_mujoco.py \
        --checkpoint /path/to/act_lerobot_224/checkpoints/030000/pretrained_model \
        --output output/act_224.mp4
"""
import argparse
import json
import os
import subprocess
import sys
import time

# Headless EGL must be set before any MuJoCo/OpenGL import
os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, load_calib_wrist, make_model_with_cube,
    get_model_ids, solve_ik, get_tcp_pose, HOME_QPOS,
)
setup_egl()

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# LeRobot ACT imports
LEROBOT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "lerobot_minho")
sys.path.insert(0, os.path.join(LEROBOT_ROOT, "src"))
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# ── Defaults ──
DEFAULT_CHECKPOINT = os.path.join(
    os.path.expanduser("~"), "DATA", "INTERN", "training",
    "act_lerobot_224", "checkpoints", "030000", "pretrained_model",
)
DEFAULT_CUBE_POS = [0.45, -0.05, 0.02]   # (x, y, z) on table
DEFAULT_CUBE_YAW_DEG = 0.0
CUBE_HALF_SIZE = (0.06, 0.02, 0.02)      # 12×4×4 cm lying flat
GRIPPER_CLOSE_THRESHOLD = 0.5
FPS = 15


def parse_args():
    p = argparse.ArgumentParser(description="ACT MuJoCo inference")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to ACT pretrained_model directory")
    p.add_argument("--cube-pos", type=float, nargs=3, default=DEFAULT_CUBE_POS,
                   metavar=("X", "Y", "Z"),
                   help="Initial cube position (world frame)")
    p.add_argument("--cube-yaw", type=float, default=DEFAULT_CUBE_YAW_DEG,
                   help="Initial cube yaw angle (degrees)")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Maximum inference steps")
    p.add_argument("--action-horizon", type=int, default=10,
                   help="Number of actions to execute per chunk before re-planning")
    p.add_argument("--ema-alpha", type=float, default=0.7,
                   help="EMA smoothing factor for pos/rot (1.0=no smoothing, lower=smoother)")
    p.add_argument("--no-gripper-ema", action="store_true",
                   help="Disable EMA on gripper (instant grip response)")
    p.add_argument("--device", default="cuda:0",
                   help="Device for model inference")
    p.add_argument("--output", default=None,
                   help="Output video path (default: output/act_infer_<timestamp>.mp4)")
    p.add_argument("--calib", default=None,
                   help="Camera calibration YAML path")
    p.add_argument("--scene-xml", default=None,
                   help="Path to fr3_with_hand.xml (auto-detected if not set)")
    return p.parse_args()


# =====================================================================
# MuJoCo environment helpers  (adapted from infer_gr00t_mujoco.py)
# =====================================================================

def init_scene(args, render_h, render_w):
    """Build MuJoCo scene with cube and return environment dict."""
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

    cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")

    renderer_base = mujoco.Renderer(model, height=render_h, width=render_w)
    renderer_wrist = mujoco.Renderer(model, height=render_h, width=render_w)

    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qposadr = model.jnt_qposadr[cube_jnt_id] if cube_jnt_id >= 0 else None

    if cube_qposadr is not None:
        data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quat_wxyz

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
    }


def get_observation_act(env):
    """Capture current observation from MuJoCo and format for ACT.

    Returns dict with:
        observation.images.cam_base: (C, H, W) float32 tensor [0, 1]
        observation.images.cam_wrist: (C, H, W) float32 tensor [0, 1]
        observation.state: (8,) float32 tensor [x,y,z, qx,qy,qz,qw, gripper_width]
    """
    model, data, ids = env["model"], env["data"], env["ids"]

    # Render cameras (RGB, uint8)
    env["renderer_base"].update_scene(data, camera=env["cam_base_id"])
    img_base = env["renderer_base"].render().copy()  # (H, W, 3) RGB uint8

    env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"])
    img_wrist = env["renderer_wrist"].render().copy()

    # EEF state
    tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
    eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()  # xyzw

    # Gripper width: normalized 0~1 (finger joint range 0..0.04)
    finger_id = ids["finger_ids"][0]
    if finger_id >= 0:
        finger_pos = data.qpos[model.jnt_qposadr[finger_id]]
        gripper_width = np.clip(finger_pos / 0.04, 0.0, 1.0)
    else:
        gripper_width = 1.0

    # State: [x, y, z, qx, qy, qz, qw, gripper_width]
    state = np.concatenate([
        tcp_pos,                          # (3,)
        eef_quat_xyzw,                    # (4,)
        np.array([gripper_width]),         # (1,)
    ]).astype(np.float32)

    # Convert images to (C, H, W) float32 [0, 1]
    img_base_t = torch.from_numpy(img_base).float() / 255.0
    img_base_t = img_base_t.permute(2, 0, 1).contiguous()  # (3, H, W)

    img_wrist_t = torch.from_numpy(img_wrist).float() / 255.0
    img_wrist_t = img_wrist_t.permute(2, 0, 1).contiguous()

    return {
        "observation.images.cam_base": img_base_t,
        "observation.images.cam_wrist": img_wrist_t,
        "observation.state": torch.from_numpy(state),
    }


def apply_action(env, eef_pos, eef_quat_xyzw, gripper_width, grasp_state):
    """Apply a single action step to MuJoCo via IK."""
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

    # Cube kinematic attachment
    cube_qposadr = env["cube_qposadr"]
    if cube_qposadr is not None:
        if not grasp_state["grasped"] and gripper_width < GRIPPER_CLOSE_THRESHOLD:
            tcp_pos_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
            T_tcp = np.eye(4)
            T_tcp[:3, :3] = tcp_R_now
            T_tcp[:3, 3] = tcp_pos_now

            cube_pos_now = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
            cube_quat_wxyz = data.qpos[cube_qposadr + 3:cube_qposadr + 7].copy()
            cube_quat_xyzw_c = [cube_quat_wxyz[1], cube_quat_wxyz[2],
                                cube_quat_wxyz[3], cube_quat_wxyz[0]]
            T_cube = np.eye(4)
            T_cube[:3, :3] = Rotation.from_quat(cube_quat_xyzw_c).as_matrix()
            T_cube[:3, 3] = cube_pos_now

            grasp_state["grasped"] = True
            grasp_state["T_cube_in_tcp"] = np.linalg.inv(T_tcp) @ T_cube

        if grasp_state["grasped"]:
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
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "act_inference")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"act_infer_{ts}.mp4")

    # ── Read config to get image resolution ──
    config_path = os.path.join(args.checkpoint, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Extract render resolution from input_features (e.g. shape [3, 224, 224] or [3, 360, 640])
    img_key = "observation.images.cam_base"
    img_shape = cfg["input_features"][img_key]["shape"]  # [C, H, W]
    render_h, render_w = img_shape[1], img_shape[2]

    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Image resolution: {render_w}x{render_h}")
    print(f"Cube pos        : {args.cube_pos}")
    print(f"Max steps       : {args.max_steps}")
    print(f"Action horizon  : {args.action_horizon}")
    print(f"EMA alpha       : {args.ema_alpha}")
    print(f"Gripper EMA     : {'OFF' if args.no_gripper_ema else 'ON'}")
    print(f"Device          : {args.device}")
    print(f"Output          : {args.output}")

    # ── Load ACT policy ──
    print("\nLoading ACT policy...")
    t0 = time.time()
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    # Override n_action_steps to match action_horizon for proper queue management
    policy.config.n_action_steps = args.action_horizon
    policy.to(args.device)
    policy.eval()
    print(f"Policy loaded in {time.time() - t0:.1f}s")
    print(f"  chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")

    # ── Load preprocessor / postprocessor ──
    print("Loading preprocessors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
    )
    print("Preprocessors ready.")

    # ── Initialize MuJoCo scene ──
    print("Initializing MuJoCo scene...")
    env = init_scene(args, render_h, render_w)
    model, data, ids = env["model"], env["data"], env["ids"]

    # IK to initial pose (above cube, pointing down)
    init_pos = np.array([args.cube_pos[0], args.cube_pos[1], 0.25])
    init_quat = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()  # point down
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             init_pos, init_quat, max_iter=500, ns_gain=1.0)
    mujoco.mj_forward(model, data)
    print("Scene ready.\n")

    # ── Inference loop ──
    grasp_state = {"grasped": False, "T_cube_in_tcp": None}
    frames_base = []
    frames_wrist = []

    # EMA state for smoothing
    ema_alpha = args.ema_alpha
    prev_pos = None
    prev_quat = None
    prev_gw = None

    # Also create a larger renderer for recording video (640x360 always)
    VID_W, VID_H = 640, 360
    renderer_vid_base = mujoco.Renderer(model, height=VID_H, width=VID_W)
    renderer_vid_wrist = mujoco.Renderer(model, height=VID_H, width=VID_W)

    print("Running inference...")
    policy.reset()

    for step in range(args.max_steps):
        # ── Get observation and query policy ──
        obs = get_observation_act(env)
        obs_batch = preprocessor(obs)

        # select_action() manages the action queue internally:
        # - calls predict_action_chunk() only when queue is empty
        # - returns one action at a time from the queue
        # This avoids chunk boundary discontinuities.
        with torch.no_grad():
            action_tensor = policy.select_action(obs_batch)  # (1, 8)

        # Postprocess: unnormalize
        action_tensor = postprocessor(action_tensor)

        act = action_tensor[0].cpu().numpy()  # (8,)
        eef_pos = act[:3].copy()
        eef_quat = act[3:7].copy()
        gw = float(act[7])

        # ── EMA smoothing to avoid jerkiness ──
        if prev_pos is not None and ema_alpha < 1.0:
            eef_pos = ema_alpha * eef_pos + (1 - ema_alpha) * prev_pos

            # Slerp for quaternion smoothing
            dot = np.dot(prev_quat, eef_quat)
            if dot < 0:
                eef_quat = -eef_quat
                dot = -dot
            if dot < 0.9995:
                # Slerp
                theta = np.arccos(np.clip(dot, -1, 1))
                sin_theta = np.sin(theta)
                w0 = np.sin((1 - ema_alpha) * theta) / sin_theta
                w1 = np.sin(ema_alpha * theta) / sin_theta
                eef_quat = w0 * prev_quat + w1 * eef_quat
            else:
                eef_quat = ema_alpha * eef_quat + (1 - ema_alpha) * prev_quat
            eef_quat = eef_quat / np.linalg.norm(eef_quat)

            if not args.no_gripper_ema:
                gw = ema_alpha * gw + (1 - ema_alpha) * prev_gw

        prev_pos = eef_pos.copy()
        prev_quat = eef_quat.copy()
        prev_gw = gw

        apply_action(env, eef_pos, eef_quat, gw, grasp_state)

        # ── Record frames (at video resolution) ──
        renderer_vid_base.update_scene(data, camera=env["cam_base_id"])
        frame_base = cv2.cvtColor(renderer_vid_base.render().copy(), cv2.COLOR_RGB2BGR)
        frames_base.append(frame_base)

        renderer_vid_wrist.update_scene(data, camera=env["cam_wrist_id"])
        frame_wrist = cv2.cvtColor(renderer_vid_wrist.render().copy(), cv2.COLOR_RGB2BGR)
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
    combined = [np.concatenate([fb, fw], axis=1) for fb, fw in zip(frames_base, frames_wrist)]
    write_video(combined, args.output, FPS)
    print("Done.")

    # Cleanup
    env["renderer_base"].close()
    env["renderer_wrist"].close()
    renderer_vid_base.close()
    renderer_vid_wrist.close()


if __name__ == "__main__":
    main()
