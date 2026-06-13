import type { EnvDefinition } from "./types";

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

const G1_ACTION_SCALE = [
  0.5475464463233948, 0.3506614565849304, 0.5475464463233948, 0.3506614565849304,
  0.4385773241519928, 0.4385773241519928, 0.5475464463233948, 0.3506614565849304,
  0.5475464463233948, 0.3506614565849304, 0.4385773241519928, 0.4385773241519928,
  0.5475464463233948, 0.4385773241519928, 0.4385773241519928, 0.4385773241519928,
  0.4385773241519928, 0.4385773241519928, 0.4385773241519928, 0.4385773241519928,
  0.07450087368488312, 0.07450087368488312, 0.4385773241519928, 0.4385773241519928,
  0.4385773241519928, 0.4385773241519928, 0.4385773241519928, 0.07450087368488312,
  0.07450087368488312,
];

// 17 * 11 = 187 height-scan rays in a 1.6 x 1.0 grid at 0.1 m spacing.
const G1_ROUGH_INPUT_SIZE = 99 + 187;

export const g1RoughEnv: EnvDefinition = {
  id: "g1_rough",
  label: "Unitree G1 Rough",
  taskId: "Mjlab-Velocity-Rough-Unitree-G1",
  sceneXmlUrl: "/envs/g1_rough/scene_optimized.xml",
  assets: [],
  observationKind: "velocity-rough-v1",
  policy: {
    onnxUrl: "/models/g1_velocity_rough_latest.onnx",
    inputSize: G1_ROUGH_INPUT_SIZE,
    outputSize: 29,
    defaultJointPos: G1_DEFAULT_JOINT_POS,
    actionScale: G1_ACTION_SCALE,
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
      linVelX: [-1.5, 2.0],
      linVelY: [-1.0, 1.0],
      angVelZ: [-0.7, 0.7],
    },
    routeFollowerMode: "holonomic",
    controlDt: 0.02,
    decimation: 4,
    terrainScan: {
      frameBody: "robot/pelvis",
      excludeBody: "robot/pelvis",
      sizeX: 1.6,
      sizeY: 1.0,
      resolution: 0.1,
      maxDistance: 5.0,
      geomGroups: [0],
    },
  },
  sim: {
    timestep: 0.005,
  },
  initialState: {
    terrainSnap: {
      jointName: "robot/floating_base_joint",
      excludeBody: "robot/pelvis",
      geomGroups: [0],
      rayStartHeight: 0.0,
      maxDistance: 5.0,
    },
  },
  render: {
    visualMeshManifestUrl: "/envs/g1_rough/render-manifest.json",
  },
  viewer: {
    followBody: "robot/torso_link",
    distance: 2.1,
    azimuthDeg: 135,
    elevationDeg: -15,
  },
};
