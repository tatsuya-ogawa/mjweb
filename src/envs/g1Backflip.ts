import type { EnvDefinition } from "./types";

// Scaffolding only: the BeyondMimic-style tracking policy requires a motion
// command (per-step motion target reference) that is not yet wired up on the
// web side. The exact observation layout for the actor head is:
//   command(motion)           [size depends on MotionCommand internals]
//   motion_anchor_pos_b       3
//   motion_anchor_ori_b       6  (6D rotation representation)
//   base_lin_vel              3
//   base_ang_vel              3
//   joint_pos                 29
//   joint_vel                 29
//   actions                   29
// Until the motion command pipeline is ported, this env definition exists so
// that the scene, assets, and ONNX wiring can be slotted in later.

const withRobotPrefix = (names: string[]) => names.map((name) => `robot/${name}`);

const G1_JOINT_NAMES = withRobotPrefix([
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]);

const G1_DEFAULT_JOINT_POS = [
  -0.31200000643730164, 0, 0, 0.6690000295639038, -0.3630000054836273, 0,
  -0.31200000643730164, 0, 0, 0.6690000295639038, -0.3630000054836273, 0,
  0, 0, 0,
  0.20000000298023224, 0.20000000298023224, 0, 0.6000000238418579, 0, 0, 0,
  0.20000000298023224, -0.20000000298023224, 0, 0.6000000238418579, 0, 0, 0,
];

// Tracking task uses JointPositionActionCfg with scale=0.5 across all joints.
const G1_BACKFLIP_ACTION_SCALE = G1_DEFAULT_JOINT_POS.map(() => 0.5);

// Placeholder. Real value depends on the trained policy's motion-command width.
const G1_BACKFLIP_INPUT_SIZE = 0;

export const g1BackflipEnv: EnvDefinition = {
  id: "g1_backflip",
  label: "Unitree G1 Backflip (scaffold)",
  taskId: "Mjlab-Tracking-Flat-Unitree-G1",
  sceneXmlUrl: "/envs/g1_backflip/scene_optimized.xml",
  assets: [],
  observationKind: "tracking-motion-v1",
  policy: {
    onnxUrl: "/models/g1_backflip.onnx",
    inputSize: G1_BACKFLIP_INPUT_SIZE,
    outputSize: 29,
    defaultJointPos: G1_DEFAULT_JOINT_POS,
    actionScale: G1_BACKFLIP_ACTION_SCALE,
    jointNames: G1_JOINT_NAMES,
    actuatorNames: G1_JOINT_NAMES,
    imuLinearVelocitySensor: "robot/imu_lin_vel",
    imuAngularVelocitySensor: "robot/imu_ang_vel",
    rootJointName: "robot/floating_base_joint",
    keyframeId: 0,
    commandDefaults: {
      linVelX: 0,
      linVelY: 0,
      angVelZ: 0,
    },
    commandLimits: {
      linVelX: [-1.0, 1.0],
      linVelY: [-1.0, 1.0],
      angVelZ: [-1.0, 1.0],
    },
    controlDt: 0.02,
    decimation: 4,
  },
  sim: {
    timestep: 0.005,
  },
  render: {
    visualMeshManifestUrl: "/envs/g1_backflip/render-manifest.json",
  },
  viewer: {
    followBody: "robot/torso_link",
    distance: 2.5,
    azimuthDeg: 135,
    elevationDeg: -10,
  },
};
