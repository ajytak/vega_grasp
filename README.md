# Vega Right-Arm Grasping with Isaac Lab

This project trains a single reinforcement learning policy for robotic grasping of multiple YCB objects using the Vega upper-body robot in Isaac Lab. The task focuses on the right arm and gripper under joint position control, with randomized object placement and optional table-height variation for robustness.

## Overview

The environment uses the Vega right arm and gripper to grasp tabletop objects from the YCB dataset. The policy is state-based: it observes robot joint states, previous action, and object pose relative to the gripper frame. The task is structured as a staged grasping problem: approach a pre-grasp waypoint, align gripper orientation, move to a near-object waypoint, close the gripper, and lift the object.

## Dependencies

- Python 3.10+
- NVIDIA GPU with CUDA
- Isaac Sim / Isaac Lab
- PyTorch
- Vega URDF assets
- YCB object assets from Isaac Sim / Isaac Nucleus

## Setup

1. Install Isaac Sim and Isaac Lab.
2. Clone this repository into your Isaac Lab workspace.
3. Ensure the Vega URDF is available at the path used in the config:
   - `dexmate/assets/robots/humanoid/vega_1u/vega_1u_gripper.urdf`
4. Ensure YCB assets are available through Isaac Nucleus.
5. Set up the Isaac Lab Python environment.
6. Register the environment/task if required.

## Usage

### Train
Launch training with the configured environment and PPO runner from your Isaac Lab setup.

Example workflow:
1. Select the environment config `DexmateEnvCfg`
2. Run the Isaac Lab training entry point with this task
3. Train across randomized resets and multiple object instances

### Debug / Visualize
Use `DexmateDebugEnv` to visualize:
- gripper frame
- object frame
- target waypoint frame

This is useful for checking reward shaping and waypoint placement.

<!-- ### Evaluate
Run the trained policy in inference mode across:
- different YCB objects
- randomized object poses
- randomized table heights if enabled

Record screen captures for the demo video. -->

## Technical Approach

- **Simulator:** Isaac Lab
- **Robot:** Vega upper body, right arm and gripper only
- **Control:** Joint position control through Isaac Lab action terms
- **Policy type:** State-based RL policy
- **Objects:** Single shared task setup across ~5 YCB objects
- **Randomization:** Object pose randomization at reset; table-height randomization supported
- **Reward design:** Multi-stage dense shaping:
  - reach pre-grasp waypoint
  - align gripper orientation with object
  - approach near-object waypoint
  - open/close gripper at the correct phase
  - lift object above table
  - penalize unstable motion and excessive velocity

## Design Decisions

- A state-based policy was used instead of vision to keep the task tractable and focus on grasp learning.
- The grasp behavior is decomposed into phases using waypoint-based reward shaping to improve exploration.
- Observations are expressed in the gripper frame where useful, making the policy more invariant to world placement.
- A single environment structure is shared across multiple YCB objects to support one policy over varied geometries.

## Repository Contents

- environment/task config
- reward and observation functions
- asset definitions for Vega and YCB objects
- debug environment for frame visualization

## Deliverables

This repository contains:
- source code for training and evaluation
- this README
- demo video showing grasping under randomized conditions

## Notes

- The current setup is right-arm only.
- Vision is not required for this implementation.
- Table-height randomization can be enabled directly from the event configuration (Table randomization only works with 1 env).