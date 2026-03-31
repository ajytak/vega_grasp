# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.utils.math import quat_apply
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp

VEGA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/workspace/isaac_projects/dexmate/source/dexmate/dexmate/assets/robots/humanoid/vega_1u/vega_1u_gripper.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    "R_arm_j[1-7]": 200.0,
                    "R_gripper_j[1-2]": 80.0,
                },
                damping={
                    "R_arm_j[1-7]": 20.0,
                    "R_gripper_j[1-2]": 8.0,
                },
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.0, 0.0, 0.15),
        rot=(0.70710678, 0.0, 0.0, 0.70710678),
        joint_pos={
            "R_arm_j1": 0.0,
            "R_arm_j2": 0.0,
            "R_arm_j3": 0.0,
            "R_arm_j4": 0.0,
            "R_arm_j5": -1.57,
            "R_arm_j6": 0.0,
            "R_arm_j7": 0.3,
            "R_gripper_j1": 0.78,
            "R_gripper_j2": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[f"R_arm_j{i}" for i in range(1, 8)],
            stiffness=200.0,
            damping=20.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["R_gripper_j1", "R_gripper_j2"],
            stiffness=80.0,
            damping=8.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)

# YCB object configurations

YCB_CRACKER_BOX_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.0, 0.75),
            rot=(0.5, -0.5, -0.5, 0.5),
        ),
    )

YCB_SUGAR_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.75),
        rot=(0.5, -0.5, -0.5, 0.5),
    ),
)

YCB_TOMATO_SOUP_CAN_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.75),
        rot=(0.5, -0.5, -0.5, 0.5),
    ),
)

YCB_MUSTARD_BOTTLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.75),
        rot=(0.5, -0.5, -0.5, 0.5),
    ),
)

YCB_GELATIN_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.12),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_angular_velocity=10.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.75),
        rot=(0.5, -0.5, -0.5, 0.5),
        ),
    )

"""Available YCB object configurations for multi-object training/evaluation."""

YCB_OBJECT_CFGS = {
    "cracker_box": YCB_CRACKER_BOX_CFG,
    "sugar_box": YCB_SUGAR_BOX_CFG,
    "tomato_soup_can": YCB_TOMATO_SOUP_CAN_CFG,
    "mustard_bottle": YCB_MUSTARD_BOTTLE_CFG,
    "gelatin_box": YCB_GELATIN_BOX_CFG,
}

@configclass
class DexmateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for right-arm tabletop grasping."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = VEGA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, 0.6)),
    )

    grasp_object = YCB_TOMATO_SOUP_CAN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object"
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.95, 0.95, 0.95), intensity=3000.0),
    )

@configclass
class ActionsCfg:
    """Joint-position action configuration for the right arm and gripper."""

    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"R_arm_j{i}" for i in range(1, 8)],
        scale=0.25,
        use_default_offset=True,
    )

    right_gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["R_gripper_j1", "R_gripper_j2"],
        scale=0.2,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        # robot proprioception
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[*[f"R_arm_j{i}" for i in range(1, 8)], "R_gripper_j1", "R_gripper_j2"]
                )
            },
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[*[f"R_arm_j{i}" for i in range(1, 8)], "R_gripper_j1", "R_gripper_j2"]
                )
            },
        )

        last_action = ObsTerm(func=mdp.last_action)

        # Express object pose in the gripper base frame to reduce dependence on world pose.
        object_pos_in_right_gripper_base = ObsTerm(
            func=mdp.object_position_in_robot_body_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["R_gripper_base"]),
                "object_cfg": SceneEntityCfg("grasp_object"),
            },
        )

        object_quat_in_right_gripper_base = ObsTerm(
            func=mdp.object_orientation_in_robot_body_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["R_gripper_base"]),
                "object_cfg": SceneEntityCfg("grasp_object"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset/randomization events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Randomize object pose on the tabletop at every reset.
    reset_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("grasp_object"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.78, 0.78),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

    # # Optional domain randomization over table height. (only works for num_envs = 1)

    # randomize_table_height = EventTerm(
    #     func=mdp.randomize_table_height_only,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("table"),
    #         "z_range": (0.55, 0.65),
    #     },
    # )


@configclass
class RewardsCfg:
    """Staged reward terms for reach, alignment, grasp closure, and lift."""

    # Phase 1: pre-grasp reach
    reach_goal = RewTerm(
        func=mdp.gripper_to_goal_waypoint_reward,
        weight=1.0,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=["R_gripper_base"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
            "object_cfg": SceneEntityCfg("grasp_object"),
            "target_height_offset": 0.19,
            "sharpness": 5.0,
        },
    )

    orient_goal = RewTerm(
        func=mdp.gripper_goal_orientation_reward,
        weight=0.5,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=["R_gripper_base"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
            "object_cfg": SceneEntityCfg("grasp_object"),
            "second_target_offset": 0.12,
            "phase2_weight": 0.1,
            "fade_sharpness": 10
        },
    )

    # Phase 2: near-object approach and open gripper
    second_reach_goal = RewTerm(
        func=mdp.gripper_to_object_second_waypoint_reward,
        weight=3.0,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=["R_gripper_base"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
            "object_cfg": SceneEntityCfg("grasp_object"),
            "first_target_offset": 0.19,
            "second_target_offset": 0.12,
            "sharpness": 10.0,
            "reach_done_threshold": 0.03,
            "orient_done_threshold": 0.85,
        },
    )

    open_gripper = RewTerm(
        func=mdp.gripper_open_reward,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["R_gripper_j1", "R_gripper_j2"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
            "object_cfg": SceneEntityCfg("grasp_object"),
            "joint_open_pos": 0.79,
            "sharpness": 10.0,
        },
    )

    # Phase 3: close gripper and lift
    close_gripper = RewTerm(
        func=mdp.gripper_close_reward,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["R_gripper_j1", "R_gripper_j2"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
            "object_cfg": SceneEntityCfg("grasp_object"),
            "second_target_offset": 0.12,
            "reach_done_threshold": 0.06,
            "joint_closed_pos": 0.0,
            "joint_open_pos": 0.79,
        },
    )

    object_lifted = RewTerm(
        func=mdp.object_height_above_table,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("grasp_object"),
            "table_cfg": SceneEntityCfg("table"),
            "lift_height": 0.0,
            "table_thickness": 0.05,
        },
    )

    # Regularization
    object_motion_after_lift_penalty = RewTerm(
        func=mdp.object_motion_after_lift_penalty,
        weight=-0.1,
        params={
            "object_cfg": SceneEntityCfg("grasp_object"),
            "table_cfg": SceneEntityCfg("table"),
            "lift_height": 0.0,
            "table_thickness": 0.05,
            "lin_vel_scale": 0.5,
            "ang_vel_scale": 0.1,
        },
    )

    arm_joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_penalty_when_near_object,
        weight=-0.1,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=[*[f"R_arm_j{i}" for i in range(1, 8)], "R_gripper_j1", "R_gripper_j2"],
            ),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l1"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["R_gripper_l2"]),
            "object_cfg": SceneEntityCfg("grasp_object"),
            "distance_threshold": 0.20,
        },
    )

    gripper_vel_penalty = RewTerm(
        func=mdp.gripper_base_velocity_penalty,
        weight=-0.2,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=["R_gripper_base"]),
            "linear_weight": 1.0,
            "angular_weight": 0.1,
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-3,
    )

    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-1e-4,
    )

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class DexmateEnvCfg(ManagerBasedRLEnvCfg):
    """Basic RL env cfg for right-arm grasping."""

    scene: DexmateSceneCfg = DexmateSceneCfg(num_envs=1024, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""

        self.decimation = 4
        self.episode_length_s = 6.0

        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.5, 0.0, 1.0)

        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.max_depenetration_velocity = 5.0

class DexmateDebugEnv(ManagerBasedRLEnv):
    """Debug environment with frame markers for gripper and target visualization."""
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/object_frame")
        frame_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.obj_frame_marker = VisualizationMarkers(frame_cfg)

        frame_cfg2 = FRAME_MARKER_CFG.replace(prim_path="/Visuals/gripper_frame")
        frame_cfg2.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.gripper_frame_marker = VisualizationMarkers(frame_cfg2)

        self._gripper_body_id = self.scene["robot"].find_bodies("R_gripper_base")[0][0]
        self.phase2_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase3_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_debug_frames(self):
        """Visualize gripper pose and current target waypoint in the world frame."""
        robot = self.scene["robot"]
        obj = self.scene["grasp_object"]

        gripper_pos_w = robot.data.body_pos_w[:, self._gripper_body_id]
        gripper_quat_w = robot.data.body_quat_w[:, self._gripper_body_id]

        obj_pos_w = obj.data.root_pos_w
        obj_quat_w = obj.data.root_quat_w

        n = obj_pos_w.shape[0]
        device = obj_pos_w.device

        ey = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(n, 1)
        obj_y_w = quat_apply(obj_quat_w, ey)
        target_height_offset = self.cfg.rewards.reach_goal.params["target_height_offset"]

        target_pos_w = obj_pos_w - target_height_offset * obj_y_w

        self.gripper_frame_marker.visualize(gripper_pos_w, gripper_quat_w)
        self.obj_frame_marker.visualize(target_pos_w, obj_quat_w)

    def _reset_idx(self, env_ids):
        """Reset per-environment phase latches."""
        super()._reset_idx(env_ids)
        self.phase2_started[env_ids] = False
        self.phase3_started[env_ids] = False

    def step(self, action):
        out = super().step(action)
        self._update_debug_frames()
        return out