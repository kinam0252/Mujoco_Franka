#!/usr/bin/env python3
"""Sweep ACT EMA configs across all episodes (no video, just success rates)."""
import os, sys, time, json, glob
import numpy as np
import pandas as pd
import torch
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
LEROBOT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "lerobot_minho")
sys.path.insert(0, os.path.join(LEROBOT_ROOT, "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from utils import (load_calib, load_calib_wrist, make_model_with_cube,
                   get_model_ids, get_tcp_pose, solve_ik, HOME_QPOS)

CUBE_HALF_SIZE = (0.06, 0.02, 0.02)
GRIPPER_CLOSE_THRESHOLD = 0.7
LIFT_SUCCESS_CM = 4.0
MAX_STEPS = 300
DATA_DIR = os.path.expanduser("~/Repos/Intern/assets/lift_data")
CHECKPOINT = os.path.expanduser("~/DATA/INTERN/training/act_640_c20/checkpoints/100000/pretrained_model")
SCENE_XML = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")

# Configs to sweep: (ema_alpha, action_horizon)
CONFIGS = [
    (0.5, 5),  (0.5, 10), (0.5, 15), (0.5, 20),
    (0.6, 5),  (0.6, 10), (0.6, 15), (0.6, 20),
    (0.7, 5),  (0.7, 10), (0.7, 15), (0.7, 20),
    (0.8, 5),  (0.8, 10), (0.8, 15), (0.8, 20),
    (0.9, 5),  (0.9, 10), (0.9, 15), (0.9, 20),
]


def extract_cube_pose(csv_path):
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
    return [float(close_pos[0]), float(close_pos[1]), 0.02], float(yaw_deg)


def run_episode(policy, preprocessor, postprocessor, cube_pos, cube_yaw_deg,
                render_h, render_w, ema_alpha, action_horizon, device, calib_path=None):
    """Run one episode, return lift_cm."""
    T_base_cam = load_calib(calib_path)
    load_calib_wrist(calib_path)

    yaw = np.radians(cube_yaw_deg)
    q_xyzw = Rotation.from_euler('z', yaw).as_quat()
    cube_quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    model = make_model_with_cube(T_base_cam, np.array(cube_pos), cube_quat_wxyz,
                                  cube_size=CUBE_HALF_SIZE, scene_xml=SCENE_XML)
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
            data.qpos[model.jnt_qposadr[fid]] = 0.04
    mujoco.mj_forward(model, data)

    # IK to initial pose
    init_pos = np.array([cube_pos[0], cube_pos[1], 0.25])
    init_quat = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()
    solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
             init_pos, init_quat, max_iter=500, ns_gain=1.0)
    mujoco.mj_forward(model, data)

    # Override action horizon
    policy.config.n_action_steps = action_horizon
    policy.reset()

    grasp_state = {"grasped": False, "T_cube_in_tcp": None}
    prev_pos = prev_quat = prev_gw = None

    for step in range(MAX_STEPS):
        # Observation
        renderer_base.update_scene(data, camera=cam_base_id)
        img_base = renderer_base.render().copy()
        renderer_wrist.update_scene(data, camera=cam_wrist_id)
        img_wrist = renderer_wrist.render().copy()

        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()
        finger_id = ids["finger_ids"][0]
        gripper_width = np.clip(data.qpos[model.jnt_qposadr[finger_id]] / 0.04, 0.0, 1.0) if finger_id >= 0 else 1.0

        state = np.concatenate([tcp_pos, eef_quat_xyzw, [gripper_width]]).astype(np.float32)
        obs = {
            "observation.images.cam_base": torch.from_numpy(img_base).float().div(255).permute(2,0,1).contiguous(),
            "observation.images.cam_wrist": torch.from_numpy(img_wrist).float().div(255).permute(2,0,1).contiguous(),
            "observation.state": torch.from_numpy(state),
        }

        obs_batch = preprocessor(obs)
        with torch.no_grad():
            action_tensor = policy.select_action(obs_batch)
        action_tensor = postprocessor(action_tensor)
        act = action_tensor[0].cpu().numpy()

        eef_pos = act[:3].copy()
        eef_quat = act[3:7].copy()
        gw = float(act[7])

        # EMA (position+rotation only, no gripper)
        if prev_pos is not None and ema_alpha < 1.0:
            eef_pos = ema_alpha * eef_pos + (1 - ema_alpha) * prev_pos
            dot = np.dot(prev_quat, eef_quat)
            if dot < 0:
                eef_quat = -eef_quat
                dot = -dot
            if dot < 0.9995:
                theta = np.arccos(np.clip(dot, -1, 1))
                sin_theta = np.sin(theta)
                w0 = np.sin((1 - ema_alpha) * theta) / sin_theta
                w1 = np.sin(ema_alpha * theta) / sin_theta
                eef_quat = w0 * prev_quat + w1 * eef_quat
            else:
                eef_quat = ema_alpha * eef_quat + (1 - ema_alpha) * prev_quat
            eef_quat /= np.linalg.norm(eef_quat)
            # NO gripper EMA

        prev_pos, prev_quat, prev_gw = eef_pos.copy(), eef_quat.copy(), gw

        # Apply action
        quat_norm = np.linalg.norm(eef_quat)
        if quat_norm > 1e-6:
            eef_quat = eef_quat / quat_norm
        solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
                 eef_pos, eef_quat, max_iter=50)
        finger_pos_val = np.clip(gw, 0.0, 1.0) * 0.04
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = finger_pos_val

        # Cube attachment
        if cube_qposadr is not None:
            if not grasp_state["grasped"] and gw < GRIPPER_CLOSE_THRESHOLD:
                tcp_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
                T_tcp = np.eye(4); T_tcp[:3,:3] = tcp_R_now; T_tcp[:3,3] = tcp_now
                cube_p = data.qpos[cube_qposadr:cube_qposadr+3].copy()
                cube_qw = data.qpos[cube_qposadr+3:cube_qposadr+7].copy()
                T_cube = np.eye(4)
                T_cube[:3,:3] = Rotation.from_quat([cube_qw[1],cube_qw[2],cube_qw[3],cube_qw[0]]).as_matrix()
                T_cube[:3,3] = cube_p
                grasp_state["grasped"] = True
                grasp_state["T_cube_in_tcp"] = np.linalg.inv(T_tcp) @ T_cube

            if grasp_state["grasped"]:
                tcp_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
                T_tcp = np.eye(4); T_tcp[:3,:3] = tcp_R_now; T_tcp[:3,3] = tcp_now
                T_cube = T_tcp @ grasp_state["T_cube_in_tcp"]
                data.qpos[cube_qposadr:cube_qposadr+3] = T_cube[:3,3]
                q_c = Rotation.from_matrix(T_cube[:3,:3]).as_quat()
                data.qpos[cube_qposadr+3:cube_qposadr+7] = [q_c[3],q_c[0],q_c[1],q_c[2]]

        mujoco.mj_forward(model, data)

    # Result
    if cube_qposadr is not None:
        lift_cm = max(0.0, (data.qpos[cube_qposadr + 2] - cube_pos[2]) * 100.0)
    else:
        lift_cm = 0.0

    # Cleanup renderers
    del renderer_base, renderer_wrist
    return lift_cm


def main():
    print("=== ACT EMA Sweep (no-gripper-ema) ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Configs: {len(CONFIGS)} (ema_alpha × action_horizon)")
    print(f"Data dir: {DATA_DIR}")
    print()

    # Load episodes
    episodes = sorted(glob.glob(os.path.join(DATA_DIR, "kinam_*")))
    ep_info = []
    for ep_dir in episodes:
        csv_path = os.path.join(ep_dir, "eef_pose_quat.csv")
        if not os.path.exists(csv_path):
            continue
        cube_pos, cube_yaw = extract_cube_pose(csv_path)
        if cube_pos is None:
            continue
        ep_info.append({"name": os.path.basename(ep_dir), "cube_pos": cube_pos, "cube_yaw": cube_yaw})
    print(f"Found {len(ep_info)} episodes\n")

    # Load model once
    config_path = os.path.join(CHECKPOINT, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    img_shape = cfg["input_features"]["observation.images.cam_base"]["shape"]
    render_h, render_w = img_shape[1], img_shape[2]
    print(f"Resolution: {render_w}x{render_h}")

    print("Loading ACT policy...")
    t0 = time.time()
    policy = ACTPolicy.from_pretrained(CHECKPOINT)
    device = "cuda:0"
    policy.to(device)
    policy.eval()
    print(f"Policy loaded in {time.time()-t0:.1f}s  chunk_size={policy.config.chunk_size}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=CHECKPOINT)

    # Results storage
    results = {cfg_key: [] for cfg_key in CONFIGS}

    total_runs = len(CONFIGS) * len(ep_info)
    run_idx = 0
    t_start = time.time()

    for ema_alpha, horizon in CONFIGS:
        cfg_name = f"ema{ema_alpha}_h{horizon}"
        successes = 0
        print(f"\n{'='*50}")
        print(f"Config: {cfg_name}")
        print(f"{'='*50}")

        for i, ep in enumerate(ep_info):
            run_idx += 1
            lift_cm = run_episode(
                policy, preprocessor, postprocessor,
                ep["cube_pos"], ep["cube_yaw"],
                render_h, render_w,
                ema_alpha, horizon, device)
            success = lift_cm >= LIFT_SUCCESS_CM
            results[(ema_alpha, horizon)].append((ep["name"], lift_cm, success))
            if success:
                successes += 1

            elapsed = time.time() - t_start
            eta = elapsed / run_idx * (total_runs - run_idx)
            status = "✓" if success else "✗"
            print(f"  [{run_idx}/{total_runs}] {ep['name']}: {lift_cm:5.1f}cm {status}  "
                  f"({successes}/{i+1})  ETA {eta/60:.0f}min")

        rate = successes / len(ep_info) * 100
        print(f"  → {cfg_name}: {successes}/{len(ep_info)} = {rate:.0f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    summary = []
    for (ema_alpha, horizon), res_list in sorted(results.items()):
        succ = sum(1 for _,_,s in res_list if s)
        rate = succ / len(res_list) * 100
        mean_lift = np.mean([l for _,l,_ in res_list])
        summary.append((rate, succ, len(res_list), ema_alpha, horizon, mean_lift))
        print(f"  ema={ema_alpha} h={horizon:>2d}:  {succ:>2d}/{len(res_list)} ({rate:5.1f}%)  mean_lift={mean_lift:.1f}cm")

    summary.sort(key=lambda x: (-x[0], -x[5]))
    print(f"\n  BEST: ema={summary[0][3]} h={summary[0][4]} → {summary[0][1]}/{summary[0][2]} ({summary[0][0]:.1f}%) mean_lift={summary[0][5]:.1f}cm")
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
