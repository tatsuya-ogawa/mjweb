import PolicyWorker from "./policyWorker?worker";
import type {
  CommandState,
  PolicyEnvDefinition,
  TerrainScanConfig,
} from "../envs/types";
import { clamp, quatRotateInverse, vectorNorm } from "./math";

type MujocoModule = any;
type MujocoModel = any;
type MujocoData = any;

interface JointAddress {
  qpos: number;
  qvel: number;
}

/**
 * Mirror torch.arange(-size/2, size/2 + resolution/2, resolution).
 * For size=1.6, resolution=0.1 this yields 17 evenly spaced points in
 * [-0.8, 0.8]; size=1.0, resolution=0.1 yields 11 points in [-0.5, 0.5].
 */
function generateGridAxis(size: number, resolution: number): Float64Array {
  const halfSize = size / 2;
  const stop = halfSize + resolution * 0.5;
  const count = Math.max(0, Math.floor((stop - -halfSize) / resolution) + 1);
  const axis = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    axis[i] = -halfSize + i * resolution;
  }
  return axis;
}

interface HeightScanState {
  frameBodyName: string;
  excludeBodyId: number;
  /** Local-frame ray offsets (X then Y, in mjlab grid order). Length = numRays. */
  offsetsX: Float64Array;
  offsetsY: Float64Array;
  numRays: number;
  maxDistance: number;
  invMaxDistance: number;
  /** vec6 with -1 for included groups, 0 for excluded. */
  geomGroup: number[];
  /** Scratch buffers reused per ray. */
  pnt: number[];
  vec: number[];
  geomidOut: Int32Array;
  normalOut: Float64Array;
}

export interface PolicyStats {
  actionNorm: number;
  lastInferenceMs: number;
  inputName: string;
  outputName: string;
}

interface PolicyWorkerMetadata {
  inputName: string;
  outputName: string;
}

interface PolicyWorkerInference {
  action: Float32Array;
  inferenceMs: number;
}

type PolicyWorkerPayload = PolicyWorkerMetadata | PolicyWorkerInference | Record<string, never>;

interface PolicyWorkerResponse {
  id: number;
  ok: boolean;
  payload?: PolicyWorkerPayload;
  error?: string;
}

interface PendingWorkerRequest {
  resolve: (payload: PolicyWorkerPayload) => void;
  reject: (error: Error) => void;
}

class PolicyWorkerClient {
  private readonly worker = new PolicyWorker();
  private readonly pending = new Map<number, PendingWorkerRequest>();
  private nextId = 1;

  constructor() {
    this.worker.onmessage = (event: MessageEvent<PolicyWorkerResponse>) => {
      const pending = this.pending.get(event.data.id);
      if (!pending) {
        return;
      }
      this.pending.delete(event.data.id);
      if (event.data.ok) {
        pending.resolve(event.data.payload ?? {});
      } else {
        pending.reject(new Error(event.data.error ?? "Policy worker request failed"));
      }
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || "Policy worker error");
      for (const pending of this.pending.values()) {
        pending.reject(error);
      }
      this.pending.clear();
    };
  }

  async init(onnxUrl: string): Promise<PolicyWorkerMetadata> {
    return this.request<PolicyWorkerMetadata>({ type: "init", onnxUrl });
  }

  async infer(observation: Float32Array): Promise<PolicyWorkerInference> {
    const transferObservation = new Float32Array(observation);
    return this.request<PolicyWorkerInference>(
      { type: "infer", observation: transferObservation },
      [transferObservation.buffer],
    );
  }

  async dispose(): Promise<void> {
    try {
      await this.request<Record<string, never>>({ type: "dispose" });
    } finally {
      this.worker.terminate();
      for (const pending of this.pending.values()) {
        pending.reject(new Error("Policy worker disposed"));
      }
      this.pending.clear();
    }
  }

  private request<T extends PolicyWorkerPayload>(
    message: Record<string, unknown>,
    transfer: Transferable[] = [],
  ): Promise<T> {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        resolve: (payload) => resolve(payload as T),
        reject,
      });
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }
}

export class PolicyController {
  readonly command: CommandState;

  private readonly env: PolicyEnvDefinition;
  private readonly mujoco: MujocoModule;
  private readonly model: MujocoModel;
  private readonly data: MujocoData;
  private readonly policyWorker: PolicyWorkerClient;
  private readonly jointAddresses: JointAddress[];
  private readonly actuatorCtrlIds: number[];
  private readonly actuatorToJointIndex: number[];
  private readonly rootQposAddress: number;
  private readonly observation: Float32Array;
  private readonly lastAction: Float32Array;
  private readonly execAction: Float32Array;
  private readonly targetJointPos: Float64Array;
  private readonly inputName: string;
  private readonly outputName: string;
  private readonly heightScan: HeightScanState | null;
  private lastInferenceMs = 0;

  private constructor(
    env: PolicyEnvDefinition,
    mujoco: MujocoModule,
    model: MujocoModel,
    data: MujocoData,
    policyWorker: PolicyWorkerClient,
    metadata: PolicyWorkerMetadata,
  ) {
    this.env = env;
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.policyWorker = policyWorker;
    this.command = { ...env.policy.commandDefaults };
    this.observation = new Float32Array(env.policy.inputSize);
    this.lastAction = new Float32Array(env.policy.outputSize);
    this.execAction = new Float32Array(env.policy.outputSize);
    this.targetJointPos = new Float64Array(env.policy.outputSize);
    this.inputName = metadata.inputName;
    this.outputName = metadata.outputName;
    this.heightScan = env.policy.terrainScan
      ? this.initHeightScan(env.policy.terrainScan)
      : null;

    const rootJoint = this.model.jnt(env.policy.rootJointName);
    this.rootQposAddress = rootJoint.qposadr;

    this.jointAddresses = env.policy.jointNames.map((name) => {
      const joint = this.model.jnt(name);
      return { qpos: joint.qposadr, qvel: joint.dofadr };
    });

    this.actuatorCtrlIds = env.policy.actuatorNames.map((name) => {
      const actuator = this.data.actuator(name);
      return actuator.id;
    });

    this.actuatorToJointIndex = env.policy.actuatorNames.map((name) => {
      const index = env.policy.jointNames.indexOf(name);
      if (index < 0) {
        throw new Error(`Actuator target is not in jointNames: ${name}`);
      }
      return index;
    });
  }

  static async create(
    env: PolicyEnvDefinition,
    mujoco: MujocoModule,
    model: MujocoModel,
    data: MujocoData,
  ): Promise<PolicyController> {
    const policyWorker = new PolicyWorkerClient();
    try {
      const metadata = await policyWorker.init(env.policy.onnxUrl);
      return new PolicyController(env, mujoco, model, data, policyWorker, metadata);
    } catch (error) {
      await policyWorker.dispose();
      throw error;
    }
  }

  reset(): void {
    this.lastAction.fill(0);
    this.execAction.fill(0);
    this.writeDefaultJointPositions();
    this.writeDefaultTargets();
    this.mujoco.mj_forward(this.model, this.data);
  }

  async dispose(): Promise<void> {
    await this.policyWorker.dispose();
  }

  setCommand(partial: Partial<CommandState>): void {
    const limits = this.env.policy.commandLimits;
    if (partial.linVelX !== undefined) {
      this.command.linVelX = clamp(partial.linVelX, limits.linVelX[0], limits.linVelX[1]);
    }
    if (partial.linVelY !== undefined) {
      this.command.linVelY = clamp(partial.linVelY, limits.linVelY[0], limits.linVelY[1]);
    }
    if (partial.angVelZ !== undefined) {
      this.command.angVelZ = clamp(partial.angVelZ, limits.angVelZ[0], limits.angVelZ[1]);
    }
  }

  zeroCommand(): void {
    this.setCommand({ linVelX: 0, linVelY: 0, angVelZ: 0 });
  }

  applyControl(): void {
    this.writeJointTargets();
  }

  async inferAndApply(): Promise<PolicyStats> {
    const obs = this.buildObservation();
    const { action, inferenceMs } = await this.policyWorker.infer(obs);
    this.lastInferenceMs = inferenceMs;
    if (action.length < this.env.policy.outputSize) {
      throw new Error(`ONNX action size ${action.length} is smaller than expected`);
    }

    for (let i = 0; i < this.env.policy.outputSize; i += 1) {
      const rawAction = Number(action[i]);
      const clippedAction =
        this.env.policy.actionClip === undefined
          ? rawAction
          : clamp(rawAction, -this.env.policy.actionClip, this.env.policy.actionClip);
      this.execAction[i] = clippedAction;
      this.lastAction[i] = clippedAction;
      this.targetJointPos[i] =
        this.env.policy.defaultJointPos[i] +
        this.execAction[i] * this.env.policy.actionScale[i];
    }

    return {
      actionNorm: vectorNorm(this.lastAction),
      lastInferenceMs: this.lastInferenceMs,
      inputName: this.inputName,
      outputName: this.outputName,
    };
  }

  private buildObservation(): Float32Array {
    if (this.env.observationKind === "tracking-motion-v1") {
      throw new Error(
        "tracking-motion-v1 observation is not yet implemented on the web; this env is scaffolding only.",
      );
    }
    let offset = 0;
    const linVel = this.data.sensor(this.env.policy.imuLinearVelocitySensor).data;
    const angVel = this.data.sensor(this.env.policy.imuAngularVelocitySensor).data;

    for (let i = 0; i < 3; i += 1) {
      this.observation[offset] = linVel[i];
      offset += 1;
    }
    for (let i = 0; i < 3; i += 1) {
      this.observation[offset] = angVel[i];
      offset += 1;
    }

    const qpos = this.data.qpos;
    const projectedGravity = quatRotateInverse(
      qpos.subarray(this.rootQposAddress + 3, this.rootQposAddress + 7),
      [0, 0, -1],
    );
    for (let i = 0; i < 3; i += 1) {
      this.observation[offset] = projectedGravity[i];
      offset += 1;
    }

    for (let i = 0; i < this.jointAddresses.length; i += 1) {
      this.observation[offset] =
        this.data.qpos[this.jointAddresses[i].qpos] - this.env.policy.defaultJointPos[i];
      offset += 1;
    }

    for (const address of this.jointAddresses) {
      this.observation[offset] = this.data.qvel[address.qvel];
      offset += 1;
    }

    for (let i = 0; i < this.lastAction.length; i += 1) {
      this.observation[offset] = this.lastAction[i];
      offset += 1;
    }

    this.observation[offset] = this.command.linVelX;
    this.observation[offset + 1] = this.command.linVelY;
    this.observation[offset + 2] = this.command.angVelZ;
    offset += 3;

    if (this.heightScan) {
      offset = this.writeHeightScan(this.heightScan, offset);
    }

    return this.observation;
  }

  private initHeightScan(cfg: TerrainScanConfig): HeightScanState {
    const xs = generateGridAxis(cfg.sizeX, cfg.resolution);
    const ys = generateGridAxis(cfg.sizeY, cfg.resolution);
    const numRays = xs.length * ys.length;
    const offsetsX = new Float64Array(numRays);
    const offsetsY = new Float64Array(numRays);
    // mjlab uses torch.meshgrid(x, y, indexing="xy") then flatten — equivalent
    // to iterating Y outer, X inner so the ray order is row-major over y.
    let i = 0;
    for (let yi = 0; yi < ys.length; yi += 1) {
      for (let xi = 0; xi < xs.length; xi += 1) {
        offsetsX[i] = xs[xi];
        offsetsY[i] = ys[yi];
        i += 1;
      }
    }
    const excludeBody = this.model.body(cfg.excludeBody);
    const geomGroup = [0, 0, 0, 0, 0, 0];
    for (const g of cfg.geomGroups) {
      if (g >= 0 && g <= 5) {
        geomGroup[g] = -1;
      }
    }
    return {
      frameBodyName: cfg.frameBody,
      excludeBodyId: excludeBody.id,
      offsetsX,
      offsetsY,
      numRays,
      maxDistance: cfg.maxDistance,
      invMaxDistance: 1 / cfg.maxDistance,
      geomGroup,
      pnt: [0, 0, 0],
      vec: [0, 0, -1],
      geomidOut: new Int32Array(1),
      normalOut: new Float64Array(3),
    };
  }

  private writeHeightScan(state: HeightScanState, startOffset: number): number {
    const body = this.data.body(state.frameBodyName);
    const px = body.xpos[0];
    const py = body.xpos[1];
    const pz = body.xpos[2];
    const xmat = body.xmat;
    // Yaw-only rotation: project body X-axis onto world XY plane, fall back to
    // the Y-axis when the X-axis is nearly vertical (matches mjlab
    // _extract_yaw_rotation behaviour).
    let cx = xmat[0];
    let cy = xmat[3];
    let cn = Math.hypot(cx, cy);
    if (cn < 0.1) {
      // Body Y-axis is (xmat[1], xmat[4], xmat[7]); take its XY projection,
      // normalise, then rotate -90 deg around Z to recover forward direction.
      let yx = xmat[1];
      let yy = xmat[4];
      const yn = Math.max(Math.hypot(yx, yy), 1e-6);
      yx /= yn;
      yy /= yn;
      cx = yy;
      cy = -yx;
      cn = 1;
    } else {
      cx /= cn;
      cy /= cn;
    }
    const cosYaw = cx;
    const sinYaw = cy;

    const pnt = state.pnt;
    const vec = state.vec;
    let offset = startOffset;
    for (let r = 0; r < state.numRays; r += 1) {
      const lx = state.offsetsX[r];
      const ly = state.offsetsY[r];
      pnt[0] = px + cosYaw * lx - sinYaw * ly;
      pnt[1] = py + sinYaw * lx + cosYaw * ly;
      pnt[2] = pz;
      const dist = this.mujoco.mj_ray(
        this.model,
        this.data,
        pnt,
        vec,
        state.geomGroup,
        1,
        state.excludeBodyId,
        state.geomidOut,
        state.normalOut,
      );
      const height = dist < 0 || dist > state.maxDistance ? state.maxDistance : dist;
      this.observation[offset] = height * state.invMaxDistance;
      offset += 1;
    }
    return offset;
  }

  private writeDefaultTargets(): void {
    for (let i = 0; i < this.env.policy.outputSize; i += 1) {
      this.targetJointPos[i] = this.env.policy.defaultJointPos[i];
    }
    this.writeJointTargets();
  }

  private writeDefaultJointPositions(): void {
    for (let i = 0; i < this.jointAddresses.length; i += 1) {
      this.data.qpos[this.jointAddresses[i].qpos] = this.env.policy.defaultJointPos[i];
      this.data.qvel[this.jointAddresses[i].qvel] = 0;
    }
  }

  private writeJointTargets(): void {
    for (let i = 0; i < this.actuatorCtrlIds.length; i += 1) {
      const jointIndex = this.actuatorToJointIndex[i];
      const actuatorId = this.actuatorCtrlIds[i];
      if (this.env.policy.jointControlMode === "ideal-pd") {
        const idealPd = this.env.policy.idealPd;
        if (!idealPd) {
          throw new Error("ideal-pd joint control requires idealPd gains.");
        }
        const address = this.jointAddresses[jointIndex];
        const target = this.targetJointPos[jointIndex];
        const torque =
          idealPd.stiffness * (target - this.data.qpos[address.qpos]) -
          idealPd.damping * this.data.qvel[address.qvel];
        this.data.ctrl[actuatorId] = this.clampActuatorControl(actuatorId, torque);
      } else {
        this.data.ctrl[actuatorId] = this.targetJointPos[jointIndex];
      }
    }
  }

  private clampActuatorControl(actuatorId: number, value: number): number {
    const limited = this.model.actuator_ctrllimited?.[actuatorId];
    const range = this.model.actuator_ctrlrange;
    if (!limited || !range) {
      return value;
    }
    return clamp(value, range[actuatorId * 2], range[actuatorId * 2 + 1]);
  }
}
