"""Shared utilities for MuJoCo Franka rendering.

Provides:
- EGL headless rendering setup
- Camera calibration loading (`.calib` / `.yaml`)
- MuJoCo scene builder with calibrated camera injection
- IK solver (damped least-squares with null-space bias)
- ROS2 bag reader (CDR parsing, zstd decompression)
"""
import os
import sqlite3
import struct
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

# ── Headless EGL setup (must run before mujoco import in scripts) ──
def setup_egl():
    """Set environment variables for headless EGL rendering."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    for p in [os.path.expanduser("~/.local/lib/gl"), os.path.expanduser("~/.local/lib")]:
        if os.path.isdir(p):
            os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            break


# ── Paths ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
DEFAULT_CALIB = os.path.join(_ROOT_DIR, "config", "camera_info_66ep.yaml")
DEFAULT_SCENE_XML = os.path.join(
    _ROOT_DIR, "mujoco_menagerie", "franka_fr3", "fr3_with_hand.xml"
)

# ── Camera intrinsics (defaults; overwritten by load_calib_wrist / YAML) ──
INTRINSICS = {
    "fx": 906.0086059570312,
    "fy": 904.6316528320312,
    "cx": 650.87451171875,
    "cy": 374.53753662109375,
    "width": 1280,
    "height": 720,
}
CAM_W = INTRINSICS["width"]
CAM_H = INTRINSICS["height"]

# ── Robot constants ──
JOINT_NAMES = [f"fr3_joint{i+1}" for i in range(7)]
FINGER_NAMES = ["finger_joint1", "finger_joint2"]
TCP_OFFSET = np.array([0.0, 0.0, 0.103])
# Original: # From teleop data first episode (original: [0.0, -0.3, 0.0, -1.57, 0.0, 1.57, 0.0])
HOME_QPOS = np.array([0.4565, 0.0954, 0.0093, -2.5607, 0.0259, 2.6229, 1.2348])
# From teleop data (first episode joint states)
HOME_QPOS = np.array([0.4565, 0.0954, 0.0093, -2.5607, 0.0259, 2.6229, 1.2348])
JOINT_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# ── Wrist camera calibration ──
# Offset in fr3_hand frame (from camera_calibration.yaml)
WRIST_CAM_POS_IN_HAND = np.array([-0.06653174566316539, -0.025199766898375955, 0.05951160976668403])
WRIST_CAM_QUAT_XYZW_IN_HAND = np.array([0.2722828037626761, 0.26626734082350434, 0.6461479553506482, 0.661405017959565])
WRIST_CAM_INTRINSICS = {
    "fx": 907.0780029296875,
    "fy": 906.123779296875,
    "cx": 647.2562255859375,
    "cy": 366.7811279296875,
    "width": 1280,
    "height": 720,
}


# =====================================================================
# Camera calibration
# =====================================================================

def load_calib(calib_path=None):
    """Load camera calibration → 4x4 T_base_optical.

    Supports:
      - `.calib` format (eye_on_base result, T_base→optical directly)
      - `.yaml` format (camera_info_66ep.yaml, cam_base_link extrinsics)
    """
    calib_path = calib_path or DEFAULT_CALIB
    with open(calib_path) as f:
        calib = yaml.safe_load(f)

    if "transform" in calib:
        t = calib["transform"]["translation"]
        q = calib["transform"]["rotation"]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        T[:3, 3] = [t["x"], t["y"], t["z"]]
        return T.astype(np.float64)

    if "extrinsics" in calib:
        cb = calib["extrinsics"]["cam_base"]
        R_link = Rotation.from_quat([
            cb["rotation_quaternion"]["x"], cb["rotation_quaternion"]["y"],
            cb["rotation_quaternion"]["z"], cb["rotation_quaternion"]["w"],
        ]).as_matrix()
        t_link = np.array([cb["translation"]["x"], cb["translation"]["y"], cb["translation"]["z"]])
        # link → optical: R_base_optical = R_base_link @ R_link_to_optical^T
        R_l2o = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        T = np.eye(4)
        T[:3, :3] = R_link @ R_l2o.T
        T[:3, 3] = t_link

        # Update base cam intrinsics if per-camera intrinsics present
        global INTRINSICS, CAM_W, CAM_H
        if "intrinsics" in calib:
            intr = calib["intrinsics"]
            bi = intr.get("cam_base", intr)  # per-camera or flat
            if "fx" in bi:
                INTRINSICS = {
                    "fx": float(bi["fx"]), "fy": float(bi["fy"]),
                    "cx": float(bi["cx"]), "cy": float(bi["cy"]),
                    "width": int(bi["width"]), "height": int(bi["height"]),
                }
                CAM_W = INTRINSICS["width"]
                CAM_H = INTRINSICS["height"]

        return T.astype(np.float64)

    raise ValueError(f"Unknown calibration format in {calib_path}")


def load_calib_wrist(calib_path=None):
    """Load wrist camera calibration from YAML, update module globals.

    Reads cam_wrist extrinsics and intrinsics from the same camera_info_66ep.yaml
    used for cam_base. Updates WRIST_CAM_POS_IN_HAND, WRIST_CAM_QUAT_XYZW_IN_HAND,
    and WRIST_CAM_INTRINSICS in-place.
    """
    global WRIST_CAM_POS_IN_HAND, WRIST_CAM_QUAT_XYZW_IN_HAND, WRIST_CAM_INTRINSICS

    calib_path = calib_path or DEFAULT_CALIB
    with open(calib_path) as f:
        calib = yaml.safe_load(f)

    if "extrinsics" not in calib or "cam_wrist" not in calib["extrinsics"]:
        print(f"[load_calib_wrist] WARN: no cam_wrist in {calib_path}, using defaults")
        return

    cw = calib["extrinsics"]["cam_wrist"]
    WRIST_CAM_POS_IN_HAND = np.array([
        cw["translation"]["x"], cw["translation"]["y"], cw["translation"]["z"]
    ], dtype=np.float64)
    WRIST_CAM_QUAT_XYZW_IN_HAND = np.array([
        cw["rotation_quaternion"]["x"], cw["rotation_quaternion"]["y"],
        cw["rotation_quaternion"]["z"], cw["rotation_quaternion"]["w"],
    ], dtype=np.float64)

    if "intrinsics" in calib:
        intr = calib["intrinsics"]
        # Support per-camera intrinsics (cam_wrist key) or flat intrinsics
        if "cam_wrist" in intr:
            wi = intr["cam_wrist"]
        else:
            wi = intr
        WRIST_CAM_INTRINSICS = {
            "fx": float(wi["fx"]), "fy": float(wi["fy"]),
            "cx": float(wi["cx"]), "cy": float(wi["cy"]),
            "width": int(wi["width"]), "height": int(wi["height"]),
        }

    print(f"[load_calib_wrist] Loaded from {os.path.basename(calib_path)}: "
          f"pos={WRIST_CAM_POS_IN_HAND}, quat={WRIST_CAM_QUAT_XYZW_IN_HAND}")


# =====================================================================
# MuJoCo scene builder
# =====================================================================

def _wrist_cam_xml():
    """Return XML snippet for wrist camera (placeholder in worldbody).

    WRIST_CAM_QUAT_XYZW_IN_HAND stores the optical frame quaternion directly
    (from eye_in_hand calibration: T_hand -> optical).
    We convert optical -> MuJoCo by flipping Y and Z: diag([1, -1, -1]).
    """
    R_optical = Rotation.from_quat(WRIST_CAM_QUAT_XYZW_IN_HAND).as_matrix()
    R_mj = R_optical @ np.diag([1.0, -1.0, -1.0])
    q = Rotation.from_matrix(R_mj).as_quat()  # xyzw
    qw = f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}"
    p = WRIST_CAM_POS_IN_HAND
    fovy_w = 2 * np.degrees(np.arctan2(
        WRIST_CAM_INTRINSICS["height"] / 2.0, WRIST_CAM_INTRINSICS["fy"]))
    return (f'<camera name="cam_wrist" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
            f'quat="{qw}" fovy="{fovy_w:.4f}"/>')


def _bind_wrist_cam(model):
    """Rebind the cam_wrist camera from worldbody to the hand body."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if cam_id >= 0 and hand_id >= 0:
        model.cam_bodyid[cam_id] = hand_id
        # pos and quat are already in hand-local frame from the XML

def make_model(T_base_cam, scene_xml=None, cam_name="cam_base"):
    """Load FR3 XML and inject a calibrated camera.

    Args:
        T_base_cam: 4x4 T_base→optical transform
        scene_xml: path to base XML (default: fr3_with_hand.xml)
        cam_name: name for the injected camera
    Returns:
        mujoco.MjModel
    """
    scene_xml = scene_xml or DEFAULT_SCENE_XML
    R_cam = T_base_cam[:3, :3]
    t_cam = T_base_cam[:3, 3]
    # Optical → MuJoCo camera convention: flip Y and Z
    R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS["fy"]))

    xml = f"""
    <mujoco model="fr3 scene">
      <include file="fr3_with_hand.xml"/>
      <statistic center="0.3 0 0.4" extent="1.0"/>
      <visual>
        <headlight diffuse="0.4 0.4 0.4" ambient="0.45 0.43 0.40" specular="0.15 0.15 0.15"/>
        <rgba haze="0.15 0.20 0.25 1"/>
        <global offwidth="{CAM_W}" offheight="{CAM_H}"/>
        <quality shadowsize="4096"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.35 0.35 0.38" rgb2="0.18 0.18 0.20"
                 width="512" height="3072"/>
        <texture type="2d" name="labfloor" builtin="flat"
                 rgb1="0.35 0.35 0.35" width="1" height="1"/>
        <material name="labfloor" texture="labfloor" texuniform="true" reflectance="0.05"/>
        <material name="dark_table" rgba="0.12 0.14 0.18 1" specular="0.3" shininess="0.1" reflectance="0.08"/>
      </asset>
      <worldbody>
        <light pos="0.3 0.0 1.8" dir="0 0 -1" directional="true"
               diffuse="0.45 0.45 0.45" ambient="0.15 0.14 0.13" specular="0.1 0.1 0.1"
               castshadow="true"/>
        <light pos="0.8 0.5 1.2" dir="-0.4 -0.3 -1" directional="false"
               diffuse="0.2 0.2 0.2" specular="0.05 0.05 0.05"/>
        <geom name="labfloor" size="3 3 0.01" type="plane" material="labfloor"
              pos="0.3 0 -0.02"/>
        <body name="table" pos="0.3 0 -0.01">
          <geom name="table_top" type="box" size="0.55 0.45 0.01"
                material="dark_table" contype="1" conaffinity="1"/>
        </body>
        <camera name="{cam_name}"
                pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
                quat="{quat_wxyz[0]:.6f} {quat_wxyz[1]:.6f} {quat_wxyz[2]:.6f} {quat_wxyz[3]:.6f}"
                fovy="{fovy:.4f}"/>
        <camera name="front" pos="1.0 0.0 0.8" xyaxes="0 1 0 -0.6 0 0.8"/>
        <camera name="side"  pos="0.0 0.8 0.6" xyaxes="-1 0 0 0 -0.6 0.8"/>
        <camera name="top"   pos="0.4 0.0 1.5" xyaxes="0 1 0 -1 0 0"/>
        {_wrist_cam_xml()}
      </worldbody>
    </mujoco>
    """
    orig_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(scene_xml)))
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(orig_dir)
    _bind_wrist_cam(model)
    return model


def make_model_with_cube(T_base_cam, cube_pos, cube_quat_wxyz=None,
                          cube_size=(0.02, 0.02, 0.06), scene_xml=None, cam_name="cam_base"):
    """Like make_model but with a white cuboid added to the scene.

    Args:
        cube_pos: (3,) xyz position in world frame
        cube_quat_wxyz: (4,) orientation as wxyz quaternion (default: identity)
        cube_size: (3,) half-sizes in xyz (default 4cm x 4cm x 12cm → half = 0.02, 0.02, 0.06)
    """
    scene_xml = scene_xml or DEFAULT_SCENE_XML
    R_cam = T_base_cam[:3, :3]
    t_cam = T_base_cam[:3, 3]
    R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS["fy"]))

    if cube_quat_wxyz is None:
        cube_quat_wxyz = [1, 0, 0, 0]
    cpos = " ".join(f"{v:.6f}" for v in cube_pos)
    cquat = " ".join(f"{v:.6f}" for v in cube_quat_wxyz)
    csz = " ".join(f"{v:.4f}" for v in cube_size)

    xml = f"""
    <mujoco model="fr3 scene with cube">
      <include file="fr3_with_hand.xml"/>
      <statistic center="0.3 0 0.4" extent="1.0"/>
      <visual>
        <headlight diffuse="0.4 0.4 0.4" ambient="0.45 0.43 0.40" specular="0.15 0.15 0.15"/>
        <rgba haze="0.15 0.20 0.25 1"/>
        <global offwidth="{CAM_W}" offheight="{CAM_H}"/>
        <quality shadowsize="4096"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.35 0.35 0.38" rgb2="0.18 0.18 0.20"
                 width="512" height="3072"/>
        <texture type="2d" name="labfloor" builtin="flat"
                 rgb1="0.35 0.35 0.35" width="1" height="1"/>
        <material name="labfloor" texture="labfloor" texuniform="true" reflectance="0.05"/>
        <material name="dark_table" rgba="0.12 0.14 0.18 1" specular="0.3" shininess="0.1" reflectance="0.08"/>
        <material name="wood_block" rgba="0.76 0.65 0.50 1" specular="0.08" shininess="0.02" reflectance="0.02"/>
      </asset>
      <worldbody>
        <light pos="0.3 0.0 1.8" dir="0 0 -1" directional="true"
               diffuse="0.45 0.45 0.45" ambient="0.15 0.14 0.13" specular="0.1 0.1 0.1"
               castshadow="true"/>
        <light pos="0.8 0.5 1.2" dir="-0.4 -0.3 -1" directional="false"
               diffuse="0.2 0.2 0.2" specular="0.05 0.05 0.05"/>
        <geom name="labfloor" size="3 3 0.01" type="plane" material="labfloor"
              pos="0.3 0 -0.02"/>
        <body name="table" pos="0.3 0 -0.01">
          <geom name="table_top" type="box" size="0.55 0.45 0.01"
                material="dark_table" contype="1" conaffinity="1"/>
        </body>
        <camera name="{cam_name}"
                pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
                quat="{quat_wxyz_cam[0]:.6f} {quat_wxyz_cam[1]:.6f} {quat_wxyz_cam[2]:.6f} {quat_wxyz_cam[3]:.6f}"
                fovy="{fovy:.4f}"/>
        <camera name="front" pos="1.0 0.0 0.8" xyaxes="0 1 0 -0.6 0 0.8"/>
        <camera name="side"  pos="0.0 0.8 0.6" xyaxes="-1 0 0 0 -0.6 0.8"/>
        <camera name="top"   pos="0.4 0.0 1.5" xyaxes="0 1 0 -1 0 0"/>
        {_wrist_cam_xml()}
        <body name="cube" pos="{cpos}" quat="{cquat}">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="{csz}" material="wood_block"
                mass="0.1" friction="1.0 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """
    orig_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(scene_xml)))
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(orig_dir)
    _bind_wrist_cam(model)
    return model


def get_model_ids(model):
    """Get commonly used joint/body IDs. Returns dict."""
    jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_NAMES]
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return {"jnt_ids": jnt_ids, "finger_ids": finger_ids, "hand_id": hand_id}


# =====================================================================
# IK solver
# =====================================================================

def get_tcp_pose(model, data, hand_id):
    """Get TCP position and rotation matrix in world frame."""
    hand_pos = data.xpos[hand_id].copy()
    hand_mat = data.xmat[hand_id].reshape(3, 3)
    tcp_pos = hand_pos + hand_mat @ TCP_OFFSET
    return tcp_pos, hand_mat.copy()


def solve_ik(model, data, hand_id, jnt_ids, target_pos, target_quat_xyzw,
             max_iter=100, tol_pos=1e-4, tol_rot=1e-3, step_size=0.5,
             q_ref=None, ns_gain=0.5):
    """Damped least-squares IK with null-space bias and joint limits.

    Returns:
        (pos_err, rot_err) — final errors in meters / radians
    """
    n_arm = len(jnt_ids)
    dof_ids = [model.jnt_dofadr[j] for j in jnt_ids]
    target_R = Rotation.from_quat(target_quat_xyzw).as_matrix()

    if q_ref is None:
        q_ref = HOME_QPOS.copy()

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        tcp_pos, tcp_R = get_tcp_pose(model, data, hand_id)

        dp = target_pos - tcp_pos
        dr = Rotation.from_matrix(target_R @ tcp_R.T).as_rotvec()

        pos_err = np.linalg.norm(dp)
        rot_err = np.linalg.norm(dr)
        if pos_err < tol_pos and rot_err < tol_rot:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, tcp_pos, hand_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])

        lam = 1e-4
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        dq = J_pinv @ np.concatenate([dp, dr])

        # Null-space bias
        q_cur = np.array([data.qpos[model.jnt_qposadr[j]] for j in jnt_ids])
        N = np.eye(n_arm) - J_pinv @ J
        dq += ns_gain * N @ (q_ref - q_cur)
        dq *= step_size

        for i, jid in enumerate(jnt_ids):
            q_new = data.qpos[model.jnt_qposadr[jid]] + dq[i]
            data.qpos[model.jnt_qposadr[jid]] = np.clip(
                q_new, JOINT_LOWER[i] + 0.01, JOINT_UPPER[i] - 0.01
            )

    mujoco.mj_forward(model, data)
    return pos_err, rot_err


def set_robot_pose(model, data, jnt_ids, finger_ids, joint_angles, gripper_pos=0.04):
    """Set robot joint angles and gripper opening."""
    for jid, angle in zip(jnt_ids, joint_angles):
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    for fid in finger_ids:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = gripper_pos
    mujoco.mj_forward(model, data)


def interpolate_pose(tcp_times, tcp_pos, tcp_quat, t_frame):
    """Interpolate TCP pose (linear + SLERP) at given timestamp."""
    idx = np.searchsorted(tcp_times, t_frame, side='right') - 1
    idx = max(0, min(idx, len(tcp_times) - 2))
    alpha = 0.0
    if tcp_times[idx + 1] > tcp_times[idx]:
        alpha = np.clip((t_frame - tcp_times[idx]) / (tcp_times[idx + 1] - tcp_times[idx]), 0, 1)
    pos = (1 - alpha) * tcp_pos[idx] + alpha * tcp_pos[idx + 1]
    r0 = Rotation.from_quat(tcp_quat[idx])
    r1 = Rotation.from_quat(tcp_quat[idx + 1])
    quat = Slerp([0, 1], Rotation.concatenate([r0, r1]))(alpha).as_quat()
    return pos, quat


# =====================================================================
# Rendering helpers
# =====================================================================

def render_camera(model, data, camera, width=None, height=None):
    """Render a single frame from named camera. Returns RGB numpy array."""
    w = width or CAM_W
    h = height or CAM_H
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=camera)
    img = renderer.render().copy()
    renderer.close()
    return img


def get_wrist_cam_world_pose(model, data, hand_id):
    """Compute wrist camera position and orientation in world frame.

    Returns (pos_world, R_world_link) where R_world_link is the rotation of
    the cam_hand_link frame (+X fwd, +Y left, +Z up) in the world frame.
    """
    hand_pos = data.xpos[hand_id].copy()
    hand_R = data.xmat[hand_id].reshape(3, 3).copy()
    cam_pos = hand_pos + hand_R @ WRIST_CAM_POS_IN_HAND
    R_hand_cam = Rotation.from_quat(WRIST_CAM_QUAT_XYZW_IN_HAND).as_matrix()
    R_cam_world = hand_R @ R_hand_cam
    return cam_pos, R_cam_world


# =====================================================================
# ROS2 bag reader
# =====================================================================

def _parse_cdr_header(data):
    off = 4
    sec = struct.unpack_from('<i', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    fid_len = struct.unpack_from('<I', data, off)[0]; off += 4
    frame_id = data[off:off + fid_len - 1].decode('utf-8', errors='replace')
    off += fid_len
    return sec, nsec, frame_id, off


def _align4(off):
    return (off + 3) & ~3


def _align8(raw_off):
    return (((raw_off - 4) + 7) & ~7) + 4


def _parse_pose_stamped(data):
    sec, nsec, _, off = _parse_cdr_header(data)
    off = _align8(off)
    vals = struct.unpack_from('<7d', data, off)
    return sec + nsec * 1e-9, np.array(vals[:3]), np.array(vals[3:7])


def _parse_joint_state(data):
    sec, nsec, _, off = _parse_cdr_header(data)
    t = sec + nsec * 1e-9
    off = _align4(off)
    n_names = struct.unpack_from('<I', data, off)[0]; off += 4
    names = []
    for _ in range(n_names):
        off = _align4(off)
        slen = struct.unpack_from('<I', data, off)[0]; off += 4
        names.append(data[off:off + slen - 1].decode('utf-8', errors='replace'))
        off += slen

    def _read_doubles(offset):
        offset = _align4(offset)
        n = struct.unpack_from('<I', data, offset)[0]; offset += 4
        if n > 0:
            offset = _align8(offset)
            vals = np.array(struct.unpack_from(f'<{n}d', data, offset))
            offset += n * 8
        else:
            vals = np.array([])
        return vals, offset

    position, off = _read_doubles(off)
    velocity, off = _read_doubles(off)
    effort, off = _read_doubles(off)
    return t, names, position, velocity, effort


def _parse_compressed_image(data):
    sec, nsec, _, off = _parse_cdr_header(data)
    t = sec + nsec * 1e-9
    off = _align4(off)
    fmt_len = struct.unpack_from('<I', data, off)[0]; off += 4
    off += fmt_len
    off = _align4(off)
    img_len = struct.unpack_from('<I', data, off)[0]; off += 4
    return t, data[off:off + img_len]


class BagReader:
    """Read ROS2 bag (sqlite3 + CDR). Handles .zstd decompression."""

    def __init__(self, bag_dir):
        bag_dir = Path(bag_dir)
        db3 = sorted(f for f in bag_dir.rglob("*.db3") if not str(f).endswith(".zstd"))
        if db3:
            self.db_path = str(db3[0])
        else:
            zstd = sorted(bag_dir.rglob("*.db3.zstd"))
            if not zstd:
                raise FileNotFoundError(f"No .db3 or .db3.zstd in {bag_dir}")
            self.db_path = self._decompress(str(zstd[0]))
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, name, type FROM topics")
        self.topics = {r[1]: (r[0], r[2]) for r in cur.fetchall()}
        conn.close()
        print(f"Bag: {self.db_path}  ({len(self.topics)} topics)")

    @staticmethod
    def _decompress(zstd_path):
        import zstandard as zstd
        out = zstd_path.replace(".db3.zstd", ".db3")
        if os.path.exists(out):
            return out
        print(f"  Decompressing {os.path.basename(zstd_path)}...")
        dctx = zstd.ZstdDecompressor()
        with open(zstd_path, 'rb') as fi, open(out, 'wb') as fo:
            dctx.copy_stream(fi, fo)
        return out

    def _fetch(self, topic):
        if topic not in self.topics:
            raise KeyError(f"Topic '{topic}' not found. Available: {list(self.topics.keys())}")
        tid = self.topics[topic][0]
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def read_pose_stamped(self, topic):
        ts, ps, qs = [], [], []
        for db_ts, raw in self._fetch(topic):
            try:
                _, p, q = _parse_pose_stamped(raw)
                ts.append(db_ts * 1e-9); ps.append(p); qs.append(q)
            except Exception:
                continue
        return np.array(ts), np.array(ps), np.array(qs)

    def read_joint_states(self, topic):
        ts, pos_all, vel_all, names = [], [], [], None
        for db_ts, raw in self._fetch(topic):
            try:
                _, n, pos, vel, _ = _parse_joint_state(raw)
                if len(pos) > 0:
                    ts.append(db_ts * 1e-9); pos_all.append(pos)
                    vel_all.append(vel if len(vel) else np.zeros_like(pos))
                    if names is None:
                        names = n
            except Exception:
                continue
        return np.array(ts), np.array(pos_all), np.array(vel_all), names or []

    def read_compressed_images(self, topic):
        result = []
        for db_ts, raw in self._fetch(topic):
            try:
                _, img_bytes = _parse_compressed_image(raw)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    result.append((db_ts * 1e-9, img))
            except Exception:
                continue
        return result
