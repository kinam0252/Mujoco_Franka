#!/usr/bin/env python3
"""Batch replay ALL episodes from CSV EEF pose data.

Reads eef_pose_quat.csv per episode, skips the first N noisy frames,
detects when the gripper closes to locate the cube grasp position,
places a white cuboid (4×4×12 cm) at that position, and renders
a side-by-side (SIM) replay video.

Uses file-based locking so multiple jobs can run independently.

Usage:
    # Debug single episode
    python batch_replay.py --data-dir /path/to/lift_data \
        --single /path/to/lift_data/kinam*20260401_150941_305

    # Batch all episodes
    python batch_replay.py --data-dir /path/to/lift_data --batch-size 5
"""
import argparse
import fcntl
import glob
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_egl, load_calib, load_calib_wrist, make_model, make_model_with_cube,
    get_model_ids, solve_ik, HOME_QPOS, CAM_W, CAM_H, DEFAULT_CALIB,
    WRIST_CAM_INTRINSICS,
)
setup_egl()

import cv2
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(_ROOT, "output", "replay_csv")
LOCK_DIR = os.path.join(_ROOT, "output", "replay_csv", ".locks")

# ── Wrist camera v2: direct MuJoCo quat (15deg tilt toward gripper) ──
def _wrist_cam_mj_quat():
    """Return MuJoCo cam_quat (wxyz) for wrist cam: 15deg tilt toward gripper."""
    a = np.radians(15)
    view = np.array([np.sin(a), 0, np.cos(a)])
    up = np.array([1.0, 0, 0])
    cam_z = -view / np.linalg.norm(view)
    cam_y = up - np.dot(up, cam_z) * cam_z
    cam_y /= np.linalg.norm(cam_y)
    cam_x = np.cross(cam_y, cam_z)
    R = np.column_stack([cam_x, cam_y, cam_z])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

WRIST_CAM_MJ_QUAT = _wrist_cam_mj_quat()
WRIST_CAM_MJ_POS = np.array([-0.08, 0.0, 0.0])


SKIP_FRAMES = 20  # skip first N frames (noisy data at episode start)

# Gripper close detection:
# gripper_width in CSV is normalized ~0-1.  Open ≈ 0.995, closed ≈ 0.5
GRIPPER_CLOSE_THRESHOLD = 0.7  # first frame below this = "closed"


def load_csv(csv_path):
    """Load eef_pose_quat.csv → DataFrame with columns
    [pos_x, pos_y, pos_z, qx, qy, qz, qw, gripper_width]."""
    return pd.read_csv(csv_path)


def find_gripper_close_frame(df, threshold=GRIPPER_CLOSE_THRESHOLD):
    """Return the index of the first frame where gripper_width < threshold,
    or None if the gripper never closes."""
    gw = df["gripper_width"].values
    mask = gw < threshold
    if not mask.any():
        return None
    return int(np.argmax(mask))


LIFT_SUCCESS_CM = 4.0  # cube must be lifted at least this many cm


def evaluate_episode(ep_dir):
    """Evaluate a single episode for lift success (no rendering needed).

    Success = gripper closed AND cube lifted >= LIFT_SUCCESS_CM above table.
    The cube starts at z=0.02 (half-height of 4cm-tall box lying flat).
    When grasped, the cube center follows the TCP, so the lift height is
    approximated by (TCP_z_max_after_grasp - table_z).

    Returns dict with evaluation results.
    """
    name = os.path.basename(ep_dir)
    csv_path = os.path.join(ep_dir, "eef_pose_quat.csv")
    result = {"episode": name, "success": False, "lift_cm": 0.0,
              "gripper_closed": False, "close_frame": None, "total_frames": 0}

    if not os.path.isfile(csv_path):
        result["error"] = "no CSV"
        return result

    df = load_csv(csv_path)
    result["total_frames"] = len(df)

    close_idx = find_gripper_close_frame(df)
    if close_idx is None:
        return result

    result["gripper_closed"] = True
    result["close_frame"] = int(close_idx)

    # Table z for cube center = 0.02m (half-height when lying flat)
    cube_table_z = 0.02

    # After grasp, cube follows the TCP.
    # The cube-in-TCP offset has a z-component, but for lift detection
    # we care about how much higher the TCP goes vs. the grasp moment.
    close_z = float(df.iloc[close_idx]["pos_z"])
    post_grasp = df.iloc[close_idx:]
    max_z = float(post_grasp["pos_z"].max())

    # Lift = how much the TCP rose after grasping
    lift_m = max_z - close_z
    lift_cm = max(0.0, lift_m * 100.0)
    result["lift_cm"] = round(lift_cm, 2)
    result["grasp_z"] = round(close_z * 100, 2)  # cm
    result["max_z"] = round(max_z * 100, 2)      # cm
    result["success"] = lift_cm >= LIFT_SUCCESS_CM

    return result


def find_pending(data_dir):
    """Return list of episode dirs that still need processing."""
    eps = sorted(glob.glob(os.path.join(data_dir, "kinam*2026*")))
    pending = []
    for ep in eps:
        name = os.path.basename(ep)
        csv_path = os.path.join(ep, "eef_pose_quat.csv")
        if not os.path.isfile(csv_path):
            continue
        vid_base = os.path.join(OUT_DIR, f"{name}_cam_base.mp4")
        vid_wrist = os.path.join(OUT_DIR, f"{name}_cam_wrist.mp4")
        if not (os.path.exists(vid_base) and os.path.exists(vid_wrist)):
            pending.append(ep)
    return pending


def try_claim(ep_dir):
    """Try to exclusively claim an episode via lock file. Returns lock fd or None."""
    name = os.path.basename(ep_dir)
    lock_path = os.path.join(LOCK_DIR, f"{name}.lock")
    try:
        fd = open(lock_path, 'w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(f"{os.getpid()}\n")
        fd.flush()
        return fd
    except (OSError, IOError):
        return None


def release_lock(fd):
    """Release a claimed lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
    except Exception:
        pass


def _write_video(ffmpeg_exe, frames, output_path, fps):
    """Write a list of BGR frames to mp4 via ffmpeg."""
    h, w = frames[0].shape[:2]
    w = w & ~1
    h = h & ~1
    tmp_path = output_path + ".tmp.mp4"
    proc = subprocess.Popen([
        ffmpeg_exe, '-y', '-loglevel', 'warning',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '23', '-preset', 'fast',
        tmp_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        stderr = proc.stderr.read()
        print(f"  ffmpeg broken pipe: {stderr.decode()[:300]}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        print(f"  ffmpeg error (rc={proc.returncode}): {stderr.decode()[:300]}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False
    os.rename(tmp_path, output_path)
    return True


def process_episode(T_base_cam, ep_dir, fps=15):
    """Process one episode: CSV → separate cam_base + cam_wrist videos."""
    name = os.path.basename(ep_dir)
    out_base = os.path.join(OUT_DIR, f"{name}_cam_base.mp4")
    out_wrist = os.path.join(OUT_DIR, f"{name}_cam_wrist.mp4")

    if os.path.exists(out_base) and os.path.exists(out_wrist):
        return True

    csv_path = os.path.join(ep_dir, "eef_pose_quat.csv")
    if not os.path.isfile(csv_path):
        print(f"  SKIP: no eef_pose_quat.csv")
        return False

    df = load_csv(csv_path)
    total_frames = len(df)
    if total_frames < SKIP_FRAMES + 2:
        print(f"  SKIP: too few frames ({total_frames})")
        return False

    print(f"  Total frames: {total_frames}, skipping first {SKIP_FRAMES}")

    # ── Detect gripper close → cube position & orientation ──
    close_idx = find_gripper_close_frame(df)
    has_cube = close_idx is not None
    cube_table_pos = None
    cube_table_quat_wxyz = None
    T_cube_in_tcp = None
    close_play_frame = None

    if has_cube:
        row_close = df.iloc[close_idx]
        close_pos = np.array([row_close["pos_x"], row_close["pos_y"],
                              row_close["pos_z"]])
        close_quat_xyzw = np.array([row_close["qx"], row_close["qy"],
                                     row_close["qz"], row_close["qw"]])

        # Compute cube lying-flat orientation from gripper x-axis
        R_grip = Rotation.from_quat(close_quat_xyzw).as_matrix()
        grip_x = R_grip[:, 0]  # gripper x-axis = cube long axis
        grip_x_horiz = np.array([grip_x[0], grip_x[1], 0.0])
        n = np.linalg.norm(grip_x_horiz)
        if n > 1e-6:
            grip_x_horiz /= n
        else:
            grip_x_horiz = np.array([1.0, 0.0, 0.0])
        yaw = np.arctan2(grip_x_horiz[1], grip_x_horiz[0])

        # Cube on table: wide face down, center at z = half-height (0.02)
        cube_table_pos = np.array([close_pos[0], close_pos[1], 0.02])
        cube_R_table = Rotation.from_euler('z', yaw).as_matrix()
        q_xyzw = Rotation.from_euler('z', yaw).as_quat()
        cube_table_quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Relative transform: cube-in-TCP at close moment (for kinematic attach)
        T_tcp_close = np.eye(4)
        T_tcp_close[:3, :3] = R_grip
        T_tcp_close[:3, 3] = close_pos
        T_cube_world = np.eye(4)
        T_cube_world[:3, :3] = cube_R_table
        T_cube_world[:3, 3] = cube_table_pos
        T_cube_in_tcp = np.linalg.inv(T_tcp_close) @ T_cube_world

        close_play_frame = close_idx - SKIP_FRAMES  # may be <0 if closed before skip
        print(f"  Gripper closes at frame {close_idx} "
              f"(gw={row_close['gripper_width']:.4f})")
        print(f"  Cube table pos: [{cube_table_pos[0]:.4f}, "
              f"{cube_table_pos[1]:.4f}, {cube_table_pos[2]:.4f}]")
        print(f"  Cube yaw: {np.degrees(yaw):.1f}°,  "
              f"close_play_frame: {close_play_frame}")
        # half-sizes (0.06, 0.02, 0.02) → 12cm x 4cm x 4cm lying flat
        model = make_model_with_cube(T_base_cam, cube_table_pos,
                                      cube_table_quat_wxyz,
                                      cube_size=(0.06, 0.02, 0.02))
    else:
        print(f"  No gripper close detected, rendering without cube")
        model = make_model(T_base_cam)

    # ── Save cube info to episode directory ──
    cube_info = {
        "has_cube": has_cube,
        "cube_table_pos": cube_table_pos.tolist() if has_cube else None,
        "cube_table_quat_wxyz": list(cube_table_quat_wxyz) if has_cube else None,
        "cube_size_half": [0.06, 0.02, 0.02],
        "gripper_close_frame": int(close_idx) if has_cube else None,
        "gripper_close_play_frame": int(close_play_frame) if has_cube and close_play_frame is not None else None,
        "yaw_deg": float(np.degrees(yaw)) if has_cube else None,
    }
    cube_info_path = os.path.join(ep_dir, "cube_info.json")
    with open(cube_info_path, "w") as f_ci:
        json.dump(cube_info, f_ci, indent=2)
    print(f"  Saved cube info → {cube_info_path}")

    data = mujoco.MjData(model)
    ids = get_model_ids(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")

    # Override wrist cam with v2 direct MuJoCo quat
    model.cam_quat[cam_wrist_id] = WRIST_CAM_MJ_QUAT
    model.cam_pos[cam_wrist_id] = WRIST_CAM_MJ_POS

    # Hide hand + arm link geoms (group 4) for wrist cam only
    hide_bodies = []
    for bname in ["hand", "fr3_link6", "fr3_link7"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            hide_bodies.append(bid)
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] in hide_bodies and model.geom_group[gi] == 2:
            model.geom_group[gi] = 4
    opt_base = mujoco.MjvOption()
    opt_base.geomgroup[3] = 0  # hide collision geoms
    opt_base.geomgroup[4] = 1  # show hand in cam_base
    opt_wrist = mujoco.MjvOption()
    opt_wrist.geomgroup[4] = 0  # hide hand in cam_wrist

    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    wrist_w, wrist_h = WRIST_CAM_INTRINSICS["width"], WRIST_CAM_INTRINSICS["height"]
    renderer_wrist = mujoco.Renderer(model, height=wrist_h, width=wrist_w)

    # Get cube freejoint qpos address (if cube exists)
    cube_qposadr = None
    if has_cube:
        cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          "cube_joint")
        if cube_jnt_id >= 0:
            cube_qposadr = model.jnt_qposadr[cube_jnt_id]

    # ── Skip first SKIP_FRAMES, replay the rest ──
    df_play = df.iloc[SKIP_FRAMES:].reset_index(drop=True)

    # Init IK at the first playback frame
    q_ref = HOME_QPOS.copy()
    mujoco.mj_resetData(model, data)
    for i, jid in enumerate(ids["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = q_ref[i]
    # Set initial cube on table
    if cube_qposadr is not None:
        data.qpos[cube_qposadr:cube_qposadr + 3] = cube_table_pos
        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_table_quat_wxyz
    mujoco.mj_forward(model, data)

    row0 = df_play.iloc[0]
    init_pos = np.array([row0["pos_x"], row0["pos_y"], row0["pos_z"]])
    init_quat = np.array([row0["qx"], row0["qy"], row0["qz"], row0["qw"]])
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             init_pos, init_quat, max_iter=500, q_ref=q_ref, ns_gain=1.0)

    # ── Render frames ──
    out_w, out_h = CAM_W // 2, CAM_H // 2
    frames_base = []
    frames_wrist = []

    for fi in range(len(df_play)):
        row = df_play.iloc[fi]
        pos = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
        quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]])

        solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
                 pos, quat, max_iter=50, q_ref=q_ref)

        # Gripper: CSV gripper_width is normalized 0..1 → finger joint 0..0.04
        gw = float(row["gripper_width"])
        finger_pos = np.clip(gw, 0.0, 1.0) * 0.04
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = finger_pos

        # ── Cube kinematic positioning ──
        if cube_qposadr is not None:
            if close_play_frame is not None and fi >= close_play_frame:
                # After grasp: cube follows TCP
                T_tcp = np.eye(4)
                T_tcp[:3, :3] = Rotation.from_quat(quat).as_matrix()
                T_tcp[:3, 3] = pos
                T_cube = T_tcp @ T_cube_in_tcp
                data.qpos[cube_qposadr:cube_qposadr + 3] = T_cube[:3, 3]
                q_c = Rotation.from_matrix(T_cube[:3, :3]).as_quat()  # xyzw
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = [
                    q_c[3], q_c[0], q_c[1], q_c[2]]  # wxyz
            else:
                # Before grasp: cube stays on table
                data.qpos[cube_qposadr:cube_qposadr + 3] = cube_table_pos
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = \
                    cube_table_quat_wxyz

        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam_id, scene_option=opt_base)
        sim_base = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
        frames_base.append(cv2.resize(sim_base, (out_w, out_h)))

        renderer_wrist.update_scene(data, camera=cam_wrist_id, scene_option=opt_wrist)
        sim_wrist = cv2.cvtColor(renderer_wrist.render().copy(), cv2.COLOR_RGB2BGR)
        frames_wrist.append(cv2.resize(sim_wrist, (out_w, out_h)))

    renderer.close()
    renderer_wrist.close()

    if not frames_base:
        print(f"  SKIP: no frames rendered")
        return False

    # ── Write separate videos via ffmpeg ──
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    n_frames = len(frames_base)

    ok_b = _write_video(ffmpeg_exe, frames_base, out_base, fps)
    ok_w = _write_video(ffmpeg_exe, frames_wrist, out_wrist, fps)

    if ok_b and ok_w:
        print(f"  ✓ {name}: {n_frames} frames → cam_base + cam_wrist")
    else:
        print(f"  WARN: base={'OK' if ok_b else 'FAIL'} wrist={'OK' if ok_w else 'FAIL'}")
    return ok_b and ok_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with episode subdirs")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--batch-size", type=int, default=5,
                    help="Episodes to claim per scan cycle")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--single", type=str, default=None,
                    help="Process single episode dir (for debugging)")
    ap.add_argument("--eval", action="store_true",
                    help="Evaluate all episodes for lift success (no rendering)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOCK_DIR, exist_ok=True)

    # ── Eval-only mode (no rendering, no calibration needed) ──
    if args.eval:
        import json
        eps = sorted(glob.glob(os.path.join(args.data_dir, "kinam*2026*")))
        print(f"=== Evaluating {len(eps)} episodes ===")
        results = []
        n_success = 0
        for ep in eps:
            r = evaluate_episode(ep)
            results.append(r)
            tag = "✓ SUCCESS" if r["success"] else "✗ FAIL"
            lift_str = f"{r['lift_cm']:.1f}cm"
            extra = ""
            if r["gripper_closed"]:
                extra = (f"  grasp_z={r.get('grasp_z',0):.1f}cm "
                         f"max_z={r.get('max_z',0):.1f}cm")
            else:
                extra = "  (gripper never closed)"
            print(f"  {tag}  lift={lift_str:>7s}  {r['episode']}{extra}")
            if r["success"]:
                n_success += 1

        print(f"\n=== Summary: {n_success}/{len(eps)} success "
              f"({100*n_success/max(len(eps),1):.0f}%) ===")

        # Save JSON
        eval_path = os.path.join(OUT_DIR, "eval_results.json")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump({"total": len(eps), "success": n_success,
                       "success_rate": round(n_success / max(len(eps), 1), 4),
                       "lift_threshold_cm": LIFT_SUCCESS_CM,
                       "episodes": results}, f, indent=2)
        print(f"Saved: {eval_path}")
        return

    # Load calibration once
    print("=== Loading calibration ===")
    T = load_calib(args.calib)
    load_calib_wrist(args.calib)

    # Single episode mode (debug)
    if args.single:
        name = os.path.basename(args.single.rstrip("/"))
        print(f"\n--- Single episode: {name} ---")
        ok = process_episode(T, args.single, args.fps)
        print(f"Result: {'OK' if ok else 'FAILED'}")
        return

    total_done = 0
    cycle = 0

    while True:
        cycle += 1
        pending = find_pending(args.data_dir)
        if not pending:
            print(f"\n=== All episodes processed! (total: {total_done} by this worker) ===")
            break

        print(f"\n=== Scan cycle {cycle}: {len(pending)} pending ===")

        # Claim a batch
        claimed = []
        locks = []
        for ep in pending:
            if len(claimed) >= args.batch_size:
                break
            fd = try_claim(ep)
            if fd is not None:
                claimed.append(ep)
                locks.append(fd)

        if not claimed:
            print("  No episodes to claim (all locked by other workers). Waiting 10s...")
            time.sleep(10)
            continue

        print(f"  Claimed {len(claimed)} episodes: {[os.path.basename(e) for e in claimed]}")

        # Process
        for ep, fd in zip(claimed, locks):
            name = os.path.basename(ep)
            print(f"\n--- Processing: {name} ---")
            try:
                ok = process_episode(T, ep, args.fps)
                if ok:
                    total_done += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
            finally:
                release_lock(fd)
                lock_path = os.path.join(LOCK_DIR, f"{name}.lock")
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass

    # Cleanup lock dir
    try:
        os.rmdir(LOCK_DIR)
    except OSError:
        pass
    print(f"\nDone! Worker processed {total_done} episodes total.")


if __name__ == "__main__":
    main()
