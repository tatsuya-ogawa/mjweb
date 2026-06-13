import type { EnvDefinition } from "./types";
import { publicUrl } from "../publicUrl";

const GO1_JOINT_NAMES = [
  "robot/FR_hip_joint",
  "robot/FR_thigh_joint",
  "robot/FR_calf_joint",
  "robot/FL_hip_joint",
  "robot/FL_thigh_joint",
  "robot/FL_calf_joint",
  "robot/RR_hip_joint",
  "robot/RR_thigh_joint",
  "robot/RR_calf_joint",
  "robot/RL_hip_joint",
  "robot/RL_thigh_joint",
  "robot/RL_calf_joint",
];

const GO1_ACTUATOR_NAMES = [
  "robot/FR_hip_joint",
  "robot/FR_thigh_joint",
  "robot/FL_hip_joint",
  "robot/FL_thigh_joint",
  "robot/RR_hip_joint",
  "robot/RR_thigh_joint",
  "robot/RL_hip_joint",
  "robot/RL_thigh_joint",
  "robot/FR_calf_joint",
  "robot/FL_calf_joint",
  "robot/RR_calf_joint",
  "robot/RL_calf_joint",
];

const GO1_DEFAULT_JOINT_POS = [
  0.1,
  0.9,
  -1.8,
  -0.1,
  0.9,
  -1.8,
  0.1,
  0.9,
  -1.8,
  -0.1,
  0.9,
  -1.8,
];

const HIP_THIGH_ACTION_SCALE = 0.37275403895515624;
const CALF_ACTION_SCALE = 0.2485019978022777;

export const go1FlatEnv: EnvDefinition = {
  id: "go1_flat",
  label: "Unitree Go1 Flat",
  taskId: "Mjlab-Velocity-Flat-Unitree-Go1",
  sceneXmlUrl: publicUrl("envs/go1_flat/scene_optimized.xml"),
  assets: [],
  observationKind: "velocity-flat-v1",
  policy: {
    onnxUrl: publicUrl("models/go1_velocity_flat_latest.onnx"),
    inputSize: 48,
    outputSize: 12,
    defaultJointPos: GO1_DEFAULT_JOINT_POS,
    actionScale: [
      HIP_THIGH_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      CALF_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      CALF_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      CALF_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      HIP_THIGH_ACTION_SCALE,
      CALF_ACTION_SCALE,
    ],
    jointNames: GO1_JOINT_NAMES,
    actuatorNames: GO1_ACTUATOR_NAMES,
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
    controlDt: 0.02,
    decimation: 4,
  },
  sim: {
    timestep: 0.005,
  },
  render: {
    visualMeshManifestUrl: publicUrl("envs/go1_flat/render-manifest.json"),
  },
  viewer: {
    followBody: "robot/trunk",
    distance: 1.5,
    azimuthDeg: 135,
    elevationDeg: -12,
  },
};
