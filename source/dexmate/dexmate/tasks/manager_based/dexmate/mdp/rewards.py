# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, quat_inv, quat_mul, quat_apply
from isaaclab.assets import Articulation, RigidObject, AssetBase


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def object_position_in_robot_body_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_object"),
) -> torch.Tensor:
    """Object position expressed in the selected robot body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if robot_cfg.body_ids is None or len(robot_cfg.body_ids) != 1:
        raise ValueError("robot_cfg must resolve to exactly one body.")

    body_id = robot_cfg.body_ids[0]

    body_pos_w = robot.data.body_pos_w[:, body_id]
    body_quat_w = robot.data.body_quat_w[:, body_id]
    obj_pos_w = obj.data.root_pos_w

    return quat_apply_inverse(body_quat_w, obj_pos_w - body_pos_w)

def object_orientation_in_robot_body_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_object"),
):
    """Object orientation expressed in the selected robot body frame.

    Returns:
        Tensor of shape (num_envs, 4) in (w, x, y, z) format.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if robot_cfg.body_ids is None or len(robot_cfg.body_ids) != 1:
        raise ValueError("robot_cfg must resolve to exactly one body.")

    body_id = robot_cfg.body_ids[0]

    body_quat_w = robot.data.body_quat_w[:, body_id]
    obj_quat_w = obj.data.root_quat_w

    return quat_mul(quat_inv(body_quat_w), obj_quat_w)

def gripper_to_goal_waypoint_reward(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_height_offset: float = 0.1,
    sharpness: float = 5.0,
) -> torch.Tensor:
    """Reward the gripper midpoint for reaching a pre-grasp waypoint offset from the object."""

    robot: Articulation = env.scene[hand_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if left_finger_cfg.body_ids is None or len(left_finger_cfg.body_ids) != 1:
        raise ValueError("left_finger_cfg must resolve to exactly one body.")
    if right_finger_cfg.body_ids is None or len(right_finger_cfg.body_ids) != 1:
        raise ValueError("right_finger_cfg must resolve to exactly one body.")

    left_id = left_finger_cfg.body_ids[0]
    right_id = right_finger_cfg.body_ids[0]

    left_pos_w = robot.data.body_pos_w[:, left_id]
    right_pos_w = robot.data.body_pos_w[:, right_id]
    gripper_mid_w = 0.5 * (left_pos_w + right_pos_w)

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    n = obj_pos_w.shape[0]
    device = obj_pos_w.device

    ey = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(n, 1)
    obj_y_w = quat_apply(obj_quat_w, ey)

    target_pos_w = obj_pos_w - target_height_offset * obj_y_w
    dist = torch.norm(gripper_mid_w - target_pos_w, dim=-1)

    reward = 1.0 - torch.tanh(sharpness * dist)

    return (~env.phase2_started).float() * reward

def gripper_goal_orientation_reward(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    second_target_offset: float = 0.125,
    phase2_weight: float = 0.5,
    fade_sharpness: float = 10.0,
) -> torch.Tensor:
    """Reward alignment between the gripper approach axis and the object grasp axis."""

    robot: Articulation = env.scene[hand_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if hand_cfg.body_ids is None or len(hand_cfg.body_ids) != 1:
        raise ValueError("hand_cfg must resolve to exactly one body.")
    if left_finger_cfg.body_ids is None or len(left_finger_cfg.body_ids) != 1:
        raise ValueError("left_finger_cfg must resolve to exactly one body.")
    if right_finger_cfg.body_ids is None or len(right_finger_cfg.body_ids) != 1:
        raise ValueError("right_finger_cfg must resolve to exactly one body.")

    hand_id = hand_cfg.body_ids[0]
    left_id = left_finger_cfg.body_ids[0]
    right_id = right_finger_cfg.body_ids[0]

    hand_quat_w = robot.data.body_quat_w[:, hand_id]
    left_pos_w = robot.data.body_pos_w[:, left_id]
    right_pos_w = robot.data.body_pos_w[:, right_id]
    gripper_mid_w = 0.5 * (left_pos_w + right_pos_w)

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    n = obj_quat_w.shape[0]
    device = obj_quat_w.device

    ey = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(n, 1)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).repeat(n, 1)

    grip_z_w = quat_apply(hand_quat_w, ez)
    obj_y_w = quat_apply(obj_quat_w, ey)

    z_align = torch.sum(grip_z_w * obj_y_w, dim=-1)

    target2_pos_w = obj_pos_w - second_target_offset * obj_y_w
    dist2 = torch.norm(gripper_mid_w - target2_pos_w, dim=-1)

    alpha = torch.exp(-fade_sharpness * dist2)
    orient_weight_phase2 = 1.0 - (1.0 - phase2_weight) * alpha

    phase2_mask = env.phase2_started.float()
    orient_weight = (1.0 - phase2_mask) * 1.0 + phase2_mask * orient_weight_phase2
    
    return z_align * orient_weight

def gripper_to_object_second_waypoint_reward(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    first_target_offset: float = 0.25,
    second_target_offset: float = 0.15,
    sharpness: float = 5.0,
    reach_done_threshold: float = 0.04,
    orient_done_threshold: float = 0.90,
) -> torch.Tensor:
    """Latch phase 2 after reach-and-align success, then reward approach to the near-object waypoint."""

    robot: Articulation = env.scene[hand_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    hand_id = hand_cfg.body_ids[0]
    left_id = left_finger_cfg.body_ids[0]
    right_id = right_finger_cfg.body_ids[0]

    hand_quat_w = robot.data.body_quat_w[:, hand_id]
    left_pos_w = robot.data.body_pos_w[:, left_id]
    right_pos_w = robot.data.body_pos_w[:, right_id]
    gripper_mid_w = 0.5 * (left_pos_w + right_pos_w)

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    n = obj_pos_w.shape[0]
    device = obj_pos_w.device

    ey = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(n, 1)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).repeat(n, 1)

    obj_y_w = quat_apply(obj_quat_w, ey)
    grip_z_w = quat_apply(hand_quat_w, ez)

    first_target_pos_w = obj_pos_w - first_target_offset * obj_y_w
    first_dist = torch.norm(gripper_mid_w - first_target_pos_w, dim=-1)
    orient_score = torch.sum(grip_z_w * obj_y_w, dim=-1)

    reached_phase1 = (first_dist < reach_done_threshold) & (orient_score > orient_done_threshold)
    # Once phase 2 starts for an environment, keep it active for the rest of the episode.
    env.phase2_started |= reached_phase1

    second_target_pos_w = obj_pos_w - second_target_offset * obj_y_w
    second_dist = torch.norm(gripper_mid_w - second_target_pos_w, dim=-1)
    second_waypoint_reward = 1.0 - torch.tanh(sharpness * second_dist)

    return env.phase2_started.float() * second_waypoint_reward

def gripper_open_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    joint_open_pos: float = 0.79,
    sharpness: float = 8.0,
) -> torch.Tensor:
    """Reward keeping the gripper open during the pre-grasp approach phase."""

    robot: Articulation = env.scene[robot_cfg.name]

    if robot_cfg.joint_ids is None or len(robot_cfg.joint_ids) != 2:
        raise ValueError("robot_cfg must resolve to exactly two gripper joints.")
    if left_finger_cfg.body_ids is None or len(left_finger_cfg.body_ids) != 1:
        raise ValueError("left_finger_cfg must resolve to exactly one body.")
    if right_finger_cfg.body_ids is None or len(right_finger_cfg.body_ids) != 1:
        raise ValueError("right_finger_cfg must resolve to exactly one body.")

    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    open_err = torch.abs(joint_pos - joint_open_pos).mean(dim=-1)
    open_reward = 1.0 - torch.tanh(sharpness * open_err)

    return env.phase2_started.float() * (~env.phase3_started).float() * open_reward

def gripper_close_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    second_target_offset: float = 0.125,
    reach_done_threshold: float = 0.06,
    joint_closed_pos: float = 0.0,
    joint_open_pos: float = 0.79,
) -> torch.Tensor:
    """Latch phase 3 near the object and reward closing the gripper."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if robot_cfg.joint_ids is None or len(robot_cfg.joint_ids) != 2:
        raise ValueError("robot_cfg must resolve to exactly two gripper joints.")
    if left_finger_cfg.body_ids is None or len(left_finger_cfg.body_ids) != 1:
        raise ValueError("left_finger_cfg must resolve to exactly one body.")
    if right_finger_cfg.body_ids is None or len(right_finger_cfg.body_ids) != 1:
        raise ValueError("right_finger_cfg must resolve to exactly one body.")

    left_id = left_finger_cfg.body_ids[0]
    right_id = right_finger_cfg.body_ids[0]

    left_pos_w = robot.data.body_pos_w[:, left_id]
    right_pos_w = robot.data.body_pos_w[:, right_id]
    gripper_mid_w = 0.5 * (left_pos_w + right_pos_w)

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    n = obj_pos_w.shape[0]
    device = obj_pos_w.device

    ey = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(n, 1)
    obj_y_w = quat_apply(obj_quat_w, ey)

    second_target_pos_w = obj_pos_w - second_target_offset * obj_y_w
    second_dist = torch.norm(gripper_mid_w - second_target_pos_w, dim=-1)

    reached_phase3 = env.phase2_started & (second_dist < reach_done_threshold)
    # Once the close phase starts, keep it active for the rest of the episode.
    env.phase3_started |= reached_phase3

    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    close_err = torch.abs(joint_pos - joint_closed_pos).mean(dim=-1)

    max_err = abs(joint_open_pos - joint_closed_pos)
    close_reward = 1.0 - close_err / max_err
    close_reward = torch.clamp(close_reward, 0.0, 1.0)

    return env.phase3_started.float() * close_reward


def object_height_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    lift_height: float = 0.2,
    table_thickness: float = 0.05,
) -> torch.Tensor:
    """Dense reward for lifting object above tabletop by a relative height.

    Returns:
        Tensor of shape (num_envs,)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    table: AssetBase = env.scene[table_cfg.name]

    obj_height = obj.data.root_pos_w[:, 2]
    table_pos_w, _ = table.get_world_poses()
    table_center_height = table_pos_w[:, 2]
    table_top_height = table_center_height + 0.5 * table_thickness
    target_height = table_top_height + lift_height

    return torch.clamp(obj_height - target_height, min=0.0)

def object_motion_after_lift_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    lift_height: float = 0.08,
    table_thickness: float = 0.05,
    lin_vel_scale: float = 1.0,
    ang_vel_scale: float = 0.2,
) -> torch.Tensor:
    """Penalty on object motion after it has been lifted above the table.

    The penalty is active only after the object is above:
        table_top_height + lift_height

    Returns:
        Tensor of shape (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    table: AssetBase = env.scene[table_cfg.name]

    obj_height = obj.data.root_pos_w[:, 2]
    table_pos_w, _ = table.get_world_poses()
    table_center_height = table_pos_w[:, 2]
    table_top_height = table_center_height + 0.5 * table_thickness
    target_height = table_top_height + lift_height

    lifted = (obj_height > target_height).float()

    lin_speed_sq = torch.sum(obj.data.root_lin_vel_w ** 2, dim=-1)
    ang_speed_sq = torch.sum(obj.data.root_ang_vel_w ** 2, dim=-1)

    penalty = lin_vel_scale * lin_speed_sq + ang_vel_scale * ang_speed_sq
    return lifted * penalty

def joint_vel_penalty_when_near_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("grasp_object"),
    distance_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize robot joint velocity when the object is near the gripper center.

    Returns:
        Tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    if left_finger_cfg.body_ids is None or len(left_finger_cfg.body_ids) != 1:
        raise ValueError("left_finger_cfg must resolve to exactly one body.")
    if right_finger_cfg.body_ids is None or len(right_finger_cfg.body_ids) != 1:
        raise ValueError("right_finger_cfg must resolve to exactly one body.")
    if robot_cfg.joint_ids is None:
        raise ValueError("robot_cfg must resolve to at least one joint.")

    left_id = left_finger_cfg.body_ids[0]
    right_id = right_finger_cfg.body_ids[0]

    left_pos_w = robot.data.body_pos_w[:, left_id]
    right_pos_w = robot.data.body_pos_w[:, right_id]
    gripper_center_w = 0.5 * (left_pos_w + right_pos_w)

    obj_pos_w = obj.data.root_pos_w
    dist = torch.linalg.norm(obj_pos_w - gripper_center_w, dim=-1)

    near_object = (dist < distance_threshold).float()

    joint_vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    vel_penalty = torch.sum(torch.square(joint_vel), dim=-1)

    return near_object * vel_penalty

def gripper_base_velocity_penalty(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    linear_weight: float = 1.0,
    angular_weight: float = 0.0,
) -> torch.Tensor:
    """Penalty on gripper base linear/angular velocity."""
    robot: Articulation = env.scene[hand_cfg.name]

    if hand_cfg.body_ids is None or len(hand_cfg.body_ids) != 1:
        raise ValueError("hand_cfg must resolve to exactly one body.")

    hand_id = hand_cfg.body_ids[0]

    lin_vel_w = robot.data.body_lin_vel_w[:, hand_id]
    ang_vel_w = robot.data.body_ang_vel_w[:, hand_id]

    lin_pen = linear_weight * torch.sum(lin_vel_w * lin_vel_w, dim=-1)
    ang_pen = angular_weight * torch.sum(ang_vel_w * ang_vel_w, dim=-1)

    return (lin_pen + ang_pen)

def randomize_table_height_only(
    env,
    env_ids,
    asset_cfg,
    z_range=(0.55, 0.65),
):
    """Randomize only the table height while preserving its xy position and orientation."""

    table = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # make indexing explicit and safe
    env_ids_torch = env_ids.to(device=env.device, dtype=torch.long)
    env_ids_cpu = env_ids_torch.cpu()

    pos_w, quat_w = table.get_world_poses()
    pos_w = pos_w.clone()
    quat_w = quat_w.clone()

    z_lo, z_hi = z_range
    pos_w[env_ids_torch, 2] = torch.rand(env_ids_torch.numel(), device=env.device) * (z_hi - z_lo) + z_lo

    table.set_world_poses(
        pos_w[env_ids_torch],
        quat_w[env_ids_torch],
        indices=env_ids_cpu,
    )