import { createIcons, icons } from "lucide";
import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import { envRegistry, findEnvDefinition } from "./envs";
import type {
  CommandLimits,
  CommandState,
  EnvDefinition,
  GaussianSplatHeightfieldConfig,
  GaussianSplatPresetDefinition,
  PolicyEnvDefinition,
  TerrainSnapInitialState,
} from "./envs/types";
import {
  prepareHeightfieldScene,
  type DynamicGaussianHeightfield,
  type GaussianHeightfieldWindow,
  type GaussianHeightfieldSource,
  type GaussianSplatVisualSource,
  type HeightfieldGenerationProgress,
} from "./sim/gaussianHeightfield";
import { PolicyController, type PolicyStats } from "./sim/policyController";
import { MujocoRenderer, type SpawnPickerMoveInput } from "./sim/mujocoRenderer";
import { yawFromQuat, wrapPi } from "./sim/math";
import {
  planHeightfieldGridPath,
  planHeightfieldPath,
  type HeightfieldPathGrid,
  type HeightfieldPathOptions,
  type PlannedHeightfieldPath,
  type Vec3,
} from "./sim/heightfieldPathPlanner";
import "./styles.css";
import "./components/manual-command-controls";
import type { ManualCommandControls } from "./components/manual-command-controls";

type MujocoModule = any;
type MujocoModel = any;
type MujocoData = any;
type GaussianSupportFillMode = NonNullable<GaussianSplatHeightfieldConfig["supportFillMode"]>;

interface RuntimeState {
  env: EnvDefinition | null;
  model: MujocoModel | null;
  data: MujocoData | null;
  renderer: MujocoRenderer | null;
  policy: PolicyController | null;
  policyStats: PolicyStats | null;
  dynamicGaussianTerrain: DynamicGaussianTerrainState | null;
  paused: boolean;
  contacts: boolean;
  meshes: boolean;
  skeleton: boolean;
  heightfield: boolean;
  speed: number;
  accumulator: number;
  frameCount: number;
  fps: number;
  lastFpsAt: number;
  loading: boolean;
}

interface SpawnOverride {
  x: number;
  y: number;
}

interface RoutePlanState {
  path: PlannedHeightfieldPath;
  waypointIndex: number;
}

type RoutePlannerStatusKind = "idle" | "busy" | "success" | "error";

const DEFAULT_ROUTE_MAX_SLOPE = 0.55;
const DEFAULT_ROUTE_MAX_STEP_HEIGHT = 0.22;
const GAUSSIAN_SUPPORT_FILL_MODES: ReadonlyArray<{
  value: GaussianSupportFillMode;
  label: string;
}> = [
  { value: "fallback", label: "Fallback" },
  { value: "nearby", label: "Nearby" },
  { value: "min", label: "Min" },
];

interface DynamicGaussianTerrainState {
  runtime: DynamicGaussianHeightfield;
  centerX: number;
  centerY: number;
  pending: boolean;
  abortController: AbortController | null;
  globalMap: GaussianHeightfieldWindow | null;
  globalMapPending: boolean;
  globalMapAbortController: AbortController | null;
}

class WebPlayApp {
  private readonly root: HTMLElement;
  private readonly state: RuntimeState = {
    env: null,
    model: null,
    data: null,
    renderer: null,
    policy: null,
    policyStats: null,
    dynamicGaussianTerrain: null,
    paused: false,
    contacts: false,
    meshes: true,
    skeleton: true,
    heightfield: true,
    speed: 1,
    accumulator: 0,
    frameCount: 0,
    fps: 0,
    lastFpsAt: performance.now(),
    loading: true,
  };

  private mujoco: MujocoModule | null = null;
  private lastFrameAt = performance.now();
  private loadToken = 0;
  private heightfieldAbortController: AbortController | null = null;
  private routePlanRequestId = 0;
  private controlsEnv: EnvDefinition | null = null;
  private readonly gaussianSources = new Map<string, GaussianHeightfieldSource>();
  private readonly activeGaussianSourceLabels = new Map<string, string>();
  private readonly gaussianScaleMultipliers = new Map<string, number>();
  private readonly gaussianPresetIds = new Map<string, string>();
  private readonly gaussianSupportFillModes = new Map<string, GaussianSupportFillMode>();
  private readonly spawnOverrides = new Map<string, SpawnOverride>();
  private readonly pressedKeys = new Set<string>();
  private spawnPickerActive = false;
  private spawnPickerWasPaused = false;
  private goalPickerActive = false;
  private goalPickerWasPaused = false;
  private routePlan: RoutePlanState | null = null;
  private routeGoal: [number, number, number] | null = null;
  private routeGoalActive = false;
  private routePlannerMaxSlope = DEFAULT_ROUTE_MAX_SLOPE;
  private routePlannerMaxStepHeight = DEFAULT_ROUTE_MAX_STEP_HEIGHT;
  private routePlannerReplanTimer: number | null = null;
  private routeFollowEnabled = false;
  private activeMode: "control" | "nav" = "control";

  constructor(root: HTMLElement) {
    this.root = root;
    this.renderShell();
    this.bindUi();
  }

  async start(): Promise<void> {
    requestAnimationFrame((time) => void this.frame(time));
    this.setLoadingState(true, "Loading MuJoCo wasm", "Preparing runtime");
    await yieldForPaint();
    this.mujoco = await loadMujoco({
      locateFile: (path: string) => (path.endsWith(".wasm") ? mujocoWasmUrl : path),
    });
    await this.loadEnvironment(envRegistry[0].id);
  }

  private async loadEnvironment(envId: string): Promise<void> {
    if (!this.mujoco) {
      return;
    }
    this.stopSpawnPicker(false);
    this.stopGoalPicker(false);
    this.clearRoutePlan(false);
    this.activeMode = "control";
    const loadToken = ++this.loadToken;
    this.heightfieldAbortController?.abort();
    const heightfieldAbortController = new AbortController();
    this.heightfieldAbortController = heightfieldAbortController;
    const env = this.envWithGaussianOverrides(findEnvDefinition(envId));
    this.controlsEnv = env;
    this.syncAllControls(env);
    this.setLoadingState(true, `Loading ${env.label}`, "Fetching scene and assets");
    this.updateControlsEnabled();
    await yieldForPaint();

    const viewport = this.requiredElement<HTMLElement>("#viewport");
    let model: MujocoModel | null = null;
    let data: MujocoData | null = null;
    let policy: PolicyController | null = null;
    let renderer: MujocoRenderer | null = null;
    let gaussianSplatVisualSource: GaussianSplatVisualSource | null = null;
    let dynamicGaussianHeightfield: DynamicGaussianHeightfield | null = null;
    let vfs: any = null;
    try {
      const [sourceXml, assetBuffers] = await Promise.all([
        fetchText(env.sceneXmlUrl),
        Promise.all(env.assets.map(async (asset) => [asset, await fetchBytes(asset.url)] as const)),
      ]);
      if (loadToken !== this.loadToken) {
        return;
      }

      let xml = sourceXml;
      if (env.heightfield) {
        this.setLoadingState(true, `Loading ${env.label}`, "Generating terrain height field");
        await yieldForPaint();
        const gaussianStateKey = this.gaussianStateKey(env);
        const preparedScene = await prepareHeightfieldScene(
          env,
          sourceXml,
          this.gaussianSources.get(gaussianStateKey),
          {
            signal: heightfieldAbortController.signal,
            onProgress: (progress) => {
              if (loadToken !== this.loadToken) {
                return;
              }
              this.setLoadingState(
                true,
                `Loading ${env.label}`,
                this.heightfieldProgressDetail(progress),
              );
            },
            userScale: this.gaussianScaleMultipliers.get(gaussianStateKey),
          },
        );
        xml = preparedScene.xml;
        gaussianSplatVisualSource = preparedScene.visualSource;
        dynamicGaussianHeightfield = preparedScene.dynamicHeightfield;

        // If transform.json scale is specified, and the user hasn't overridden the scale yet,
        // sync the scale multiplier and UI controls to match the loaded scale.
        if (
          preparedScene.transformConfig &&
          typeof preparedScene.transformConfig.scale === "number" &&
          !this.gaussianScaleMultipliers.has(gaussianStateKey)
        ) {
          const loadedScale = preparedScene.transformConfig.scale;
          this.gaussianScaleMultipliers.set(gaussianStateKey, loadedScale);
          if (this.controlsEnv?.id === env.id) {
            this.syncGaussianScaleControl(env);
          }
        }

        if (preparedScene.visualSource) {
          const source = preparedScene.visualSource.source;
          this.activeGaussianSourceLabels.set(
            gaussianStateKey,
            `${source.name} / ${formatBytes(source.bytes.byteLength)}`,
          );
        } else {
          this.activeGaussianSourceLabels.delete(gaussianStateKey);
        }
        if (this.controlsEnv?.id === env.id) {
          this.syncGaussianSourceControls(env);
        }
        if (loadToken !== this.loadToken) {
          return;
        }
      }

      this.setLoadingState(true, `Loading ${env.label}`, "Preparing MuJoCo assets");
      await yieldForPaint();

      vfs = new this.mujoco.MjVFS();
      for (const [asset, bytes] of assetBuffers) {
        vfs.addBuffer(asset.vfsPath, bytes);
        const basename = asset.vfsPath.split("/").pop();
        if (basename && basename !== asset.vfsPath) {
          vfs.addBuffer(basename, bytes);
        }
      }

      this.setLoadingState(true, `Loading ${env.label}`, "Compiling MuJoCo model");
      await yieldForPaint();

      model = this.mujoco.MjModel.from_xml_string(xml, vfs);
      model.opt.timestep = env.sim.timestep;

      this.setLoadingState(true, `Loading ${env.label}`, "Initializing simulation state");
      await yieldForPaint();

      data = new this.mujoco.MjData(model);
      this.mujoco.mj_resetDataKeyframe(model, data, env.policy?.keyframeId ?? 0);
      this.applyInitialStateOverrides(env, model, data);
      this.applySpawnOverride(env, model, data);
      this.snapSpawnToTerrain(env, model, data);
      this.mujoco.mj_forward(model, data);

      if (env.policy) {
        this.setLoadingState(true, `Loading ${env.label}`, "Creating ONNX session");
        await yieldForPaint();

        policy = await PolicyController.create(
          env as PolicyEnvDefinition,
          this.mujoco,
          model,
          data,
        );
        policy.reset();
        if (loadToken !== this.loadToken) {
          return;
        }
      }

      this.setLoadingState(true, `Loading ${env.label}`, "Preparing renderer and visual meshes");
      await yieldForPaint();

      await this.disposeCurrentEnvironment();
      renderer = new MujocoRenderer(this.mujoco, viewport);
      await renderer.load(env, model, data, gaussianSplatVisualSource);
      if (loadToken !== this.loadToken) {
        return;
      }
      renderer.setMeshVisualization(this.state.meshes);
      renderer.setHeightfieldVisualization(this.state.heightfield);
      renderer.setSkeletonVisualization(this.state.skeleton);

      this.state.env = env;
      this.state.model = model;
      this.state.data = data;
      this.state.renderer = renderer;
      this.state.policy = policy;
      this.state.dynamicGaussianTerrain = dynamicGaussianHeightfield
        ? {
            runtime: dynamicGaussianHeightfield,
            centerX: dynamicGaussianHeightfield.initialCenterX,
            centerY: dynamicGaussianHeightfield.initialCenterY,
            pending: false,
            abortController: null,
            globalMap: null,
            globalMapPending: false,
            globalMapAbortController: null,
          }
        : null;
      if (this.state.dynamicGaussianTerrain) {
        this.publishDynamicGaussianTerrainDebug(this.state.dynamicGaussianTerrain);
      }
      if (policy) {
        this.syncCommandInputs(policy.command);
      } else {
        this.syncCommandInputs(null);
      }
      this.state.paused = !env.policy;
      model = null;
      data = null;
      policy = null;
      renderer = null;
      this.state.policyStats = null;
      this.state.accumulator = 0;
      this.syncAllControls(env);
      this.setLoadingState(false, this.runtimeStatus());
      this.updateControlsEnabled();
      this.updatePolicyControls();
      this.syncPauseButton();
      this.syncVisualizationButtons();
    } catch (error) {
      if (isAbortError(error) && loadToken !== this.loadToken) {
        return;
      }
      this.setLoadingState(false, error instanceof Error ? error.message : String(error));
      this.updateControlsEnabled();
      throw error;
    } finally {
      if (this.heightfieldAbortController === heightfieldAbortController) {
        this.heightfieldAbortController = null;
      }
      renderer?.dispose();
      await policy?.dispose();
      data?.delete();
      model?.delete();
      vfs?.delete();
    }
  }

  private async disposeCurrentEnvironment(): Promise<void> {
    this.state.dynamicGaussianTerrain?.abortController?.abort();
    this.state.dynamicGaussianTerrain?.globalMapAbortController?.abort();
    await this.state.policy?.dispose();
    this.state.data?.delete();
    this.state.model?.delete();
    this.state.renderer?.dispose();
    this.state.env = null;
    this.state.model = null;
    this.state.data = null;
    this.state.renderer = null;
    this.state.policy = null;
    this.state.policyStats = null;
    this.state.dynamicGaussianTerrain = null;
    this.publishDynamicGaussianTerrainDebug(null);
  }

  private async frame(time: number): Promise<void> {
    const dt = Math.min(0.05, Math.max(0, (time - this.lastFrameAt) / 1000));
    this.lastFrameAt = time;

    try {
      if (this.spawnPickerActive || this.goalPickerActive) {
        this.updateSpawnPicker(dt);
      }
      if (this.routeFollowEnabled && !this.state.paused && !this.state.loading) {
        this.updateRouteFollower();
      }
      if (!this.state.loading) {
        this.requestDynamicGaussianTerrainUpdate();
      }
      if (!this.state.paused && !this.state.loading) {
        await this.stepSimulation(dt * this.state.speed);
      }
      this.state.renderer?.update();
      this.state.renderer?.render();
      this.updateStats(time);
    } catch (error) {
      this.state.paused = true;
      this.setStatus(error instanceof Error ? error.message : String(error));
      this.syncPauseButton();
    }

    requestAnimationFrame((nextTime) => void this.frame(nextTime));
  }

  private async stepSimulation(dt: number): Promise<void> {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const policy = this.state.policy;
    if (!env || !model || !data || !this.mujoco) {
      return;
    }

    this.state.accumulator += dt;
    if (!policy || !env.policy) {
      const maxPhysicsStepsPerFrame = 8;
      let physicsSteps = 0;
      while (this.state.accumulator >= env.sim.timestep && physicsSteps < maxPhysicsStepsPerFrame) {
        this.mujoco.mj_step(model, data);
        this.state.accumulator -= env.sim.timestep;
        physicsSteps += 1;
      }
      if (physicsSteps === maxPhysicsStepsPerFrame) {
        this.state.accumulator = 0;
      }
      return;
    }

    const maxControlStepsPerFrame = 4;
    let steps = 0;
    while (this.state.accumulator >= env.policy.controlDt && steps < maxControlStepsPerFrame) {
      this.state.policyStats = await policy.inferAndApply();
      for (let i = 0; i < env.policy.decimation; i += 1) {
        policy.applyControl();
        this.mujoco.mj_step(model, data);
      }
      this.state.accumulator -= env.policy.controlDt;
      steps += 1;
    }
    if (steps === maxControlStepsPerFrame) {
      this.state.accumulator = 0;
    }
  }

  private requestDynamicGaussianTerrainUpdate(): void {
    const terrain = this.state.dynamicGaussianTerrain;
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const renderer = this.state.renderer;
    if (!terrain || !env || !model || !data || !renderer || !this.mujoco || terrain.pending) {
      return;
    }
    const rootPosition = this.currentRootPosition(env, model, data);
    if (!rootPosition) {
      return;
    }
    const dx = rootPosition[0] - terrain.centerX;
    const dy = rootPosition[1] - terrain.centerY;
    if (Math.hypot(dx, dy) < terrain.runtime.updateDistance) {
      return;
    }

    const targetX = rootPosition[0];
    const targetY = rootPosition[1];
    const abortController = new AbortController();
    terrain.pending = true;
    terrain.abortController = abortController;
    this.setStatus("Updating terrain");
    void terrain.runtime.generateWindow(targetX, targetY, {
      signal: abortController.signal,
      onProgress: (progress) => {
        if (this.state.dynamicGaussianTerrain !== terrain || abortController.signal.aborted) {
          return;
        }
        this.setStatus(this.dynamicTerrainProgressStatus(progress));
      },
    }).then((window) => {
      if (this.state.dynamicGaussianTerrain !== terrain || abortController.signal.aborted) {
        return;
      }
      terrain.centerX = window.centerX;
      terrain.centerY = window.centerY;
      this.applyDynamicGaussianTerrainWindow(window, model, data, renderer);
      this.setStatus(this.runtimeStatus());
    }).catch((error) => {
      if (!isAbortError(error)) {
        console.error(error);
        this.setStatus(error instanceof Error ? error.message : String(error));
      }
    }).finally(() => {
      if (this.state.dynamicGaussianTerrain === terrain) {
        terrain.pending = false;
        terrain.abortController = null;
      }
    });
  }

  private applyDynamicGaussianTerrainWindow(
    window: GaussianHeightfieldWindow,
    model: MujocoModel,
    data: MujocoData,
    renderer: MujocoRenderer,
  ): void {
    const hfield = model.hfield("gs_heightfield");
    const terrainGeom = model.geom("terrain");
    const adr = hfield.adr;
    const count = window.config.nrow * window.config.ncol;
    for (let i = 0; i < count; i += 1) {
      model.hfield_data[adr + i] = window.heights[i] / window.config.elevationScale;
    }
    terrainGeom.pos[0] = window.centerX;
    terrainGeom.pos[1] = window.centerY;
    terrainGeom.pos[2] = 0;
    this.mujoco?.mj_forward(model, data);
    renderer.refreshMujocoGeometry();
    this.publishDynamicGaussianTerrainDebug(this.state.dynamicGaussianTerrain);
  }

  private dynamicTerrainProgressStatus(progress: HeightfieldGenerationProgress): string {
    if (progress.stage === "support-tiles") {
      return `Terrain tiles ${progress.completed}/${progress.total}`;
    }
    if (progress.stage === "sampling") {
      return `Terrain rows ${progress.completed}/${progress.total}`;
    }
    return "Updating terrain";
  }

  private routePlanningProgressStatus(progress: HeightfieldGenerationProgress): string {
    if (progress.stage === "support-tiles") {
      return `Route map tiles ${progress.completed}/${progress.total}`;
    }
    if (progress.stage === "sampling") {
      return `Route map rows ${progress.completed}/${progress.total}`;
    }
    return "Expanding route map";
  }

  private globalMapProgressStatus(progress: HeightfieldGenerationProgress): string {
    if (progress.stage === "support-tiles") {
      return `Global map tiles ${progress.completed}/${progress.total}`;
    }
    if (progress.stage === "sampling") {
      return `Global map rows ${progress.completed}/${progress.total}`;
    }
    return "Building global map";
  }

  private publishDynamicGaussianTerrainDebug(terrain: DynamicGaussianTerrainState | null): void {
    if (!terrain) {
      delete document.documentElement.dataset.mjwebDynamicTerrain;
      return;
    }
    document.documentElement.dataset.mjwebDynamicTerrain = JSON.stringify({
      centerX: terrain.centerX,
      centerY: terrain.centerY,
      pending: terrain.pending,
      globalMapPending: terrain.globalMapPending,
      updateDistance: terrain.runtime.updateDistance,
      hfield: {
        nrow: terrain.runtime.config.nrow,
        ncol: terrain.runtime.config.ncol,
        sizeX: terrain.runtime.config.sizeX,
        sizeY: terrain.runtime.config.sizeY,
      },
      globalMap: terrain.globalMap
        ? {
            centerX: terrain.globalMap.centerX,
            centerY: terrain.globalMap.centerY,
            nrow: terrain.globalMap.config.nrow,
            ncol: terrain.globalMap.config.ncol,
            sizeX: terrain.globalMap.config.sizeX,
            sizeY: terrain.globalMap.config.sizeY,
          }
        : null,
    });
  }

  private applyInitialStateOverrides(
    env: EnvDefinition,
    model: MujocoModel,
    data: MujocoData,
  ): void {
    for (const freeJoint of env.initialState?.freeJoints ?? []) {
      const joint = model.jnt(freeJoint.jointName);
      for (let i = 0; i < freeJoint.qpos.length; i += 1) {
        data.qpos[joint.qposadr + i] = freeJoint.qpos[i];
      }
      if (freeJoint.qvel) {
        for (let i = 0; i < freeJoint.qvel.length; i += 1) {
          data.qvel[joint.dofadr + i] = freeJoint.qvel[i];
        }
      }
    }
    if (env.initialState?.terrainSnap) {
      this.mujoco.mj_forward(model, data);
      this.snapFreeJointToTerrain(env.initialState.terrainSnap, model, data);
    }
  }

  private snapFreeJointToTerrain(
    snap: TerrainSnapInitialState,
    model: MujocoModel,
    data: MujocoData,
  ): void {
    if (!this.mujoco) {
      return;
    }
    const joint = model.jnt(snap.jointName);
    const qposAddress = joint.qposadr;
    const pnt = [
      data.qpos[qposAddress],
      data.qpos[qposAddress + 1],
      data.qpos[qposAddress + 2] + snap.rayStartHeight,
    ];
    const vec = [0, 0, -1];
    const geomGroup = [0, 0, 0, 0, 0, 0];
    for (const group of snap.geomGroups) {
      if (group >= 0 && group <= 5) {
        geomGroup[group] = -1;
      }
    }
    const bodyExclude = snap.excludeBody ? model.body(snap.excludeBody).id : -1;
    const dist = this.mujoco.mj_ray(
      model,
      data,
      pnt,
      vec,
      geomGroup,
      1,
      bodyExclude,
      new Int32Array(1),
      new Float64Array(3),
    );
    if (dist >= 0 && dist <= snap.maxDistance) {
      const terrainZ = pnt[2] - dist;
      data.qpos[qposAddress + 2] += terrainZ;
    }
  }

  private reset(): void {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    if (!env || !model || !data || !this.mujoco) {
      return;
    }
    this.mujoco.mj_resetDataKeyframe(model, data, env.policy?.keyframeId ?? 0);
    this.applyInitialStateOverrides(env, model, data);
    this.applySpawnOverride(env, model, data);
    this.snapSpawnToTerrain(env, model, data);
    this.mujoco.mj_forward(model, data);
    this.state.policy?.reset();
    this.state.accumulator = 0;
    this.setStatus(this.runtimeStatus());
  }

  private startSpawnPicker(): void {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const renderer = this.state.renderer;
    if (!env || !model || !data || !renderer || !this.canUseSpawnPicker(env)) {
      return;
    }
    this.stopGoalPicker(false);
    const position = this.currentRootPosition(env, model, data);
    if (!position) {
      this.setStatus("Spawn unavailable");
      return;
    }
    this.spawnPickerWasPaused = this.state.paused;
    this.spawnPickerActive = true;
    this.pressedKeys.clear();
    this.state.paused = true;
    renderer.setSpawnPicker(true, position, { color: 0xffd166, emissive: 0x6a3f00 });
    this.setStatus(this.runtimeStatus());
    this.syncPauseButton();
    this.syncAllControls(env);
  }

  private confirmSpawnPicker(): void {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const renderer = this.state.renderer;
    if (!env || !model || !data || !renderer || !this.mujoco) {
      return;
    }
    const [x, y] = renderer.getSpawnPickerPosition();
    this.spawnOverrides.set(env.id, { x, y });
    this.mujoco.mj_resetDataKeyframe(model, data, env.policy?.keyframeId ?? 0);
    this.applyInitialStateOverrides(env, model, data);
    this.applySpawnOverride(env, model, data);
    this.snapSpawnToTerrain(env, model, data);
    this.mujoco.mj_forward(model, data);
    this.state.policy?.reset();
    this.state.accumulator = 0;
    this.clearRoutePlan(false);
    this.state.paused = this.spawnPickerWasPaused;
    this.stopSpawnPicker();
  }

  private cancelSpawnPicker(): void {
    this.state.paused = this.spawnPickerWasPaused;
    this.stopSpawnPicker();
  }

  private stopSpawnPicker(updateStatus = true): void {
    this.spawnPickerActive = false;
    this.pressedKeys.clear();
    this.state.renderer?.setSpawnPicker(false);
    this.syncAllControls(this.controlsEnv ?? this.state.env);
    this.syncPauseButton();
    if (updateStatus) {
      this.setStatus(this.runtimeStatus());
    }
  }

  private startGoalPicker(): void {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const renderer = this.state.renderer;
    if (!env || !model || !data || !renderer || !this.canUseRoutePlanner(env)) {
      return;
    }
    this.stopSpawnPicker(false);
    const root = this.currentRootPosition(env, model, data);
    if (!root) {
      this.setRoutePlannerStatus("Goal unavailable", "error");
      return;
    }
    const yaw = this.currentRootYaw(env, model, data);
    const position: [number, number, number] = [
      root[0] + Math.cos(yaw) * 3,
      root[1] + Math.sin(yaw) * 3,
      root[2],
    ];
    this.goalPickerWasPaused = this.state.paused;
    this.goalPickerActive = true;
    this.pressedKeys.clear();
    this.state.paused = true;
    renderer.setSpawnPicker(true, position, { color: 0x5bd6c6, emissive: 0x0c5a52 });
    this.setStatus(this.runtimeStatus());
    this.syncPauseButton();
    this.syncAllControls(env);
  }

  private confirmGoalPicker(): void {
    const renderer = this.state.renderer;
    if (!renderer) {
      return;
    }
    const goal = renderer.getSpawnPickerPosition();
    this.finishGoalPicker(goal);
  }

  private finishGoalPicker(goal: readonly [number, number, number]): void {
    void this.planRouteToGoal(goal);
    this.state.paused = this.goalPickerWasPaused;
    this.stopGoalPicker();
  }

  private cancelGoalPicker(): void {
    this.state.paused = this.goalPickerWasPaused;
    this.stopGoalPicker();
  }

  private stopGoalPicker(updateStatus = true): void {
    this.goalPickerActive = false;
    this.pressedKeys.clear();
    if (!this.spawnPickerActive) {
      this.state.renderer?.setSpawnPicker(false);
    }
    this.syncAllControls(this.controlsEnv ?? this.state.env);
    this.syncPauseButton();
    if (updateStatus) {
      this.setStatus(this.runtimeStatus());
    }
  }

  private async buildGlobalRouteMap(): Promise<void> {
    const terrain = this.state.dynamicGaussianTerrain;
    if (!terrain || terrain.globalMapPending || this.state.loading) {
      return;
    }
    terrain.globalMapAbortController?.abort();
    const controller = new AbortController();
    terrain.globalMapPending = true;
    terrain.globalMapAbortController = controller;
    terrain.globalMap = null;
    this.clearRoutePlan(false);
    this.setRoutePlannerStatus("Building global map", "busy");
    this.syncAllControls(this.state.env);
    try {
      const map = await terrain.runtime.generateGlobalPlanningWindow({
        signal: controller.signal,
        onProgress: (progress) => {
          if (this.state.dynamicGaussianTerrain !== terrain || controller.signal.aborted) {
            return;
          }
          this.setRoutePlannerStatus(this.globalMapProgressStatus(progress), "busy");
        },
      });
      if (this.state.dynamicGaussianTerrain !== terrain || controller.signal.aborted) {
        return;
      }
      terrain.globalMap = map;
      this.setRoutePlannerStatus(`Global map ready: ${map.config.ncol}x${map.config.nrow}`, "success");
    } catch (error) {
      if (!isAbortError(error)) {
        this.setRoutePlannerStatus(error instanceof Error ? error.message : String(error), "error");
      }
    } finally {
      if (this.state.dynamicGaussianTerrain === terrain) {
        terrain.globalMapPending = false;
        terrain.globalMapAbortController = null;
        this.syncAllControls(this.state.env);
        this.publishDynamicGaussianTerrainDebug(terrain);
      }
    }
  }

  private async planRouteToGoal(goal: readonly [number, number, number]): Promise<void> {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const renderer = this.state.renderer;
    const requestId = ++this.routePlanRequestId;
    if (!env || !model || !data || !renderer || !this.canUseRoutePlanner(env)) {
      return;
    }
    this.routeGoalActive = true;
    this.routeGoal = [goal[0], goal[1], goal[2]];
    this.routeFollowEnabled = false;
    this.routePlan = null;
    renderer.setRouteLine([]);
    renderer.setRouteGoalMarker(goal);
    const root = this.currentRootPosition(env, model, data);
    if (!root) {
      renderer.setRouteGoalMarker(goal, { color: 0xff6b5f, emissive: 0x6b1f17 });
      this.setRoutePlannerStatus("Route unavailable", "error");
      this.syncAllControls(env);
      return;
    }
    this.setRoutePlannerStatus(`Planning route (${this.routePlannerLimitsLabel()})`, "busy");
    const plannerOptions = {
      maxSlope: this.routePlannerMaxSlope,
      maxStepHeight: this.routePlannerMaxStepHeight,
      footprintRadius: 0.26,
      waypointSpacing: 0.45,
      maxExpanded: 180_000,
    };
    const path = await this.planRoutePath(
      model,
      [root[0], root[1]],
      [goal[0], goal[1]],
      plannerOptions,
      requestId,
    );
    if (requestId !== this.routePlanRequestId) {
      return;
    }
    if (!path || path.points.length < 2) {
      this.routeFollowEnabled = false;
      this.routePlan = null;
      renderer.setRouteLine([]);
      renderer.setRouteGoalMarker(goal, { color: 0xff6b5f, emissive: 0x6b1f17 });
      this.setRoutePlannerStatus(`No route available (${this.routePlannerLimitsLabel()})`, "error");
      this.syncAllControls(env);
      return;
    }
    this.routePlan = {
      path,
      waypointIndex: 1,
    };
    this.routeFollowEnabled = false;
    renderer.setRouteGoalMarker(goal);
    renderer.setRouteLine(path.points);
    this.setRoutePlannerStatus(
      `Route ready: ${path.points.length} pts / ${path.expanded} cells (${this.routePlannerLimitsLabel()})`,
      "success",
    );
    this.syncAllControls(env);
  }

  private async planRoutePath(
    model: MujocoModel,
    start: readonly [number, number],
    goal: readonly [number, number],
    options: HeightfieldPathOptions,
    requestId: number,
  ): Promise<PlannedHeightfieldPath | null> {
    const terrain = this.state.dynamicGaussianTerrain;
    if (!terrain) {
      return planHeightfieldPath(model, start, goal, options);
    }
    if (terrain.globalMap) {
      const globalWindow = terrainWindowBounds(
        terrain.globalMap.centerX,
        terrain.globalMap.centerY,
        terrain.globalMap.config.sizeX,
        terrain.globalMap.config.sizeY,
      );
      if (pointInsideBounds(start, globalWindow) && pointInsideBounds(goal, globalWindow)) {
        this.setRoutePlannerStatus("Planning on global map", "busy");
        return planHeightfieldGridPath(
          heightfieldWindowToPlannerGrid(terrain.globalMap),
          start,
          goal,
          optionsForPlanningWindow(options, terrain.globalMap),
        );
      }
    }
    const localWindow = terrainWindowBounds(
      terrain.centerX,
      terrain.centerY,
      terrain.runtime.config.sizeX,
      terrain.runtime.config.sizeY,
    );
    if (pointInsideBounds(start, localWindow) && pointInsideBounds(goal, localWindow)) {
      return planHeightfieldPath(model, start, goal, options);
    }

    this.setRoutePlannerStatus("Expanding route map", "busy");
    const controller = new AbortController();
    const window = await terrain.runtime.generateRoutePlanningWindow(start, goal, {
      signal: controller.signal,
      onProgress: (progress) => {
        if (requestId !== this.routePlanRequestId) {
          controller.abort();
          return;
        }
        this.setRoutePlannerStatus(this.routePlanningProgressStatus(progress), "busy");
      },
    });
    if (requestId !== this.routePlanRequestId) {
      return null;
    }
    return planHeightfieldGridPath(
      heightfieldWindowToPlannerGrid(window),
      start,
      goal,
      optionsForPlanningWindow(options, window),
    );
  }

  private toggleRouteFollower(): void {
    if (!this.routePlan || !this.state.policy) {
      return;
    }
    this.routeFollowEnabled = !this.routeFollowEnabled;
    if (this.routeFollowEnabled) {
      this.routePlan.waypointIndex = Math.min(1, this.routePlan.path.points.length - 1);
      this.state.paused = false;
      this.setStatus(this.runtimeStatus());
      this.syncPauseButton();
    } else {
      this.state.policy.zeroCommand();
      this.syncCommandInputs(this.state.policy.command);
      this.setStatus(this.runtimeStatus());
    }
    this.syncAllControls(this.state.env);
  }

  private clearRoutePlan(updateUi = true): void {
    this.cancelRoutePlannerReplan();
    this.routePlanRequestId += 1;
    this.routeFollowEnabled = false;
    this.routePlan = null;
    this.routeGoal = null;
    this.routeGoalActive = false;
    this.state.renderer?.setRouteLine([]);
    this.state.renderer?.setRouteGoalMarker(null);
    this.setRoutePlannerStatus("");
    if (this.state.policy) {
      this.state.policy.zeroCommand();
      this.syncCommandInputs(this.state.policy.command);
    }
    if (updateUi) {
      this.syncAllControls(this.controlsEnv ?? this.state.env);
      this.setStatus(this.runtimeStatus());
    }
  }

  private updateRouteFollower(): void {
    const env = this.state.env;
    const model = this.state.model;
    const data = this.state.data;
    const policy = this.state.policy;
    const route = this.routePlan;
    if (!env || !model || !data || !policy || !route || route.path.points.length < 2) {
      return;
    }

    const root = this.currentRootPosition(env, model, data);
    if (!root) {
      return;
    }

    // Scale tolerances and lookahead by active environment scale factor
    const scale = env.heightfield?.kind === "gaussian-splat" ? this.gaussianScaleForEnv(env) : 1.0;
    const waypointTolerance = 0.45 * scale;
    const arrivedTolerance = 0.45 * scale;
    const lookaheadDistance = 1.2 * scale;

    const points = route.path.points;
    while (
      route.waypointIndex < points.length - 1 &&
      distance2d(root, points[route.waypointIndex]) < waypointTolerance
    ) {
      route.waypointIndex += 1;
    }

    const goal = points[points.length - 1];
    if (distance2d(root, goal) < arrivedTolerance) {
      this.routeFollowEnabled = false;
      policy.zeroCommand();
      this.syncCommandInputs(policy.command);
      this.setRoutePlannerStatus("Arrived", "success");
      this.setStatus(this.runtimeStatus());
      this.syncAllControls(env);
      return;
    }

    const target = this.lookaheadRoutePoint(route.path.points, route.waypointIndex, root, lookaheadDistance);
    const yaw = this.currentRootYaw(env, model, data);
    policy.setCommand(routeFollowerCommand(env, root[0], root[1], yaw, target[0], target[1], scale));
    this.syncCommandInputs(policy.command);
  }

  private lookaheadRoutePoint(
    points: readonly Vec3[],
    startIndex: number,
    root: readonly [number, number, number],
    lookahead: number,
  ): Vec3 {
    let distanceLeft = lookahead;
    let previous: Vec3 = [root[0], root[1], root[2]];
    for (let i = startIndex; i < points.length; i += 1) {
      const point = points[i];
      const segmentLength = distance2d(previous, point);
      if (segmentLength >= distanceLeft) {
        const t = distanceLeft / Math.max(1e-6, segmentLength);
        return [
          previous[0] + (point[0] - previous[0]) * t,
          previous[1] + (point[1] - previous[1]) * t,
          previous[2] + (point[2] - previous[2]) * t,
        ];
      }
      distanceLeft -= segmentLength;
      previous = point;
    }
    return points[points.length - 1];
  }

  private updateSpawnPicker(dt: number): void {
    const renderer = this.state.renderer;
    if (!renderer) {
      return;
    }
    let forward = 0;
    let right = 0;
    let up = 0;
    if (this.pressedKeys.has("KeyW")) {
      forward += 1;
    }
    if (this.pressedKeys.has("KeyS")) {
      forward -= 1;
    }
    if (this.pressedKeys.has("KeyD")) {
      right += 1;
    }
    if (this.pressedKeys.has("KeyA")) {
      right -= 1;
    }
    if (this.pressedKeys.has("Space") || this.pressedKeys.has("KeyE")) {
      up += 1;
    }
    if (this.pressedKeys.has("KeyQ") || this.pressedKeys.has("KeyC")) {
      up -= 1;
    }
    const horizontalLength = Math.hypot(forward, right);
    if (horizontalLength > 1) {
      forward /= horizontalLength;
      right /= horizontalLength;
    }
    const fast = this.pressedKeys.has("ShiftLeft") || this.pressedKeys.has("ShiftRight");
    const input: SpawnPickerMoveInput = {
      forward,
      right,
      up,
      speed: fast ? 8.0 : 3.0,
    };
    renderer.moveSpawnPicker(input, dt);
  }

  private handleSpawnPickerKeyDown(event: KeyboardEvent): boolean {
    if (event.code === "Enter") {
      event.preventDefault();
      this.confirmSpawnPicker();
      return true;
    }
    if (event.code === "Escape") {
      event.preventDefault();
      this.cancelSpawnPicker();
      return true;
    }
    if (isSpawnPickerMovementCode(event.code)) {
      event.preventDefault();
      this.pressedKeys.add(event.code);
      return true;
    }
    return false;
  }

  private handleGoalPickerKeyDown(event: KeyboardEvent): boolean {
    if (event.code === "Enter") {
      event.preventDefault();
      this.confirmGoalPicker();
      return true;
    }
    if (event.code === "Escape") {
      event.preventDefault();
      this.cancelGoalPicker();
      return true;
    }
    if (isSpawnPickerMovementCode(event.code)) {
      event.preventDefault();
      this.pressedKeys.add(event.code);
      return true;
    }
    return false;
  }

  private applySpawnOverride(env: EnvDefinition, model: MujocoModel, data: MujocoData): void {
    const override = this.spawnOverrides.get(env.id);
    if (!override || !this.mujoco) {
      return;
    }
    const jointName = this.spawnJointName(env);
    if (!jointName) {
      return;
    }
    const joint = model.jnt(jointName);
    const qposAddress = joint.qposadr;
    const baseHeight = this.spawnBaseHeight(env, data.qpos[qposAddress + 2]);
    data.qpos[qposAddress] = override.x;
    data.qpos[qposAddress + 1] = override.y;
    data.qpos[qposAddress + 2] = baseHeight;
  }

  private snapSpawnToTerrain(env: EnvDefinition, model: MujocoModel, data: MujocoData): void {
    if (!env.heightfield || !this.mujoco) {
      return;
    }
    const jointName = this.spawnJointName(env);
    if (!jointName) {
      return;
    }
    const joint = model.jnt(jointName);
    const qposAddress = joint.qposadr;
    const x = data.qpos[qposAddress];
    const y = data.qpos[qposAddress + 1];
    const baseHeight = this.spawnBaseHeight(env, data.qpos[qposAddress + 2]);
    this.mujoco.mj_forward(model, data);
    const terrainZ = this.terrainHeightAt(env, model, data, x, y);
    if (terrainZ === null) {
      return;
    }
    const minimumZ = terrainZ + baseHeight + 0.02;
    if (data.qpos[qposAddress + 2] < minimumZ) {
      data.qpos[qposAddress + 2] = minimumZ;
    }
  }

  private currentRootPosition(
    env: EnvDefinition,
    model: MujocoModel,
    data: MujocoData,
  ): [number, number, number] | null {
    const jointName = this.spawnJointName(env);
    if (!jointName) {
      return null;
    }
    const joint = model.jnt(jointName);
    const qposAddress = joint.qposadr;
    return [
      data.qpos[qposAddress],
      data.qpos[qposAddress + 1],
      data.qpos[qposAddress + 2],
    ];
  }

  private currentRootYaw(env: EnvDefinition, model: MujocoModel, data: MujocoData): number {
    const jointName = this.spawnJointName(env);
    if (!jointName) {
      return 0;
    }
    const joint = model.jnt(jointName);
    return yawFromQuat(data.qpos.subarray(joint.qposadr + 3, joint.qposadr + 7));
  }

  private terrainHeightAt(
    env: EnvDefinition,
    model: MujocoModel,
    data: MujocoData,
    x: number,
    y: number,
  ): number | null {
    if (!this.mujoco) {
      return null;
    }
    const pnt = [x, y, Math.max(25, this.spawnBaseHeight(env, 0) + 10)];
    const vec = [0, 0, -1];
    const geomGroup = [0, 0, 0, 0, 0, 0];
    const groups = env.policy?.terrainScan?.geomGroups ?? env.initialState?.terrainSnap?.geomGroups ?? [0];
    for (const group of groups) {
      if (group >= 0 && group <= 5) {
        geomGroup[group] = -1;
      }
    }
    const bodyExclude = this.bodyIdForName(model, env.policy?.terrainScan?.excludeBody ?? env.viewer.followBody);
    const dist = this.mujoco.mj_ray(
      model,
      data,
      pnt,
      vec,
      geomGroup,
      1,
      bodyExclude,
      new Int32Array(1),
      new Float64Array(3),
    );
    if (dist < 0 || dist > 100) {
      return null;
    }
    return pnt[2] - dist;
  }

  private bodyIdForName(model: MujocoModel, bodyName: string | undefined): number {
    if (!bodyName) {
      return -1;
    }
    try {
      return model.body(bodyName).id;
    } catch {
      return -1;
    }
  }

  private spawnJointName(env: EnvDefinition): string | null {
    return env.policy?.rootJointName ?? env.initialState?.terrainSnap?.jointName ?? null;
  }

  private spawnBaseHeight(env: EnvDefinition, fallback: number): number {
    return env.heightfield?.robotBaseHeight ?? fallback;
  }

  private canUseSpawnPicker(env: EnvDefinition | null): boolean {
    return Boolean(env?.heightfield && this.spawnJointName(env));
  }

  private canUseRoutePlanner(env: EnvDefinition | null): boolean {
    return Boolean(env?.heightfield && env.policy && this.spawnJointName(env));
  }

  private renderShell(): void {
    this.root.innerHTML = `
      <div class="app-shell">
        <main id="viewport" class="viewport"></main>
        <div id="loading-overlay" class="loading-overlay is-visible" aria-live="polite">
          <div class="loading-card">
            <span class="loading-spinner" aria-hidden="true"></span>
            <span class="loading-copy">
              <strong id="loading-title">Loading</strong>
              <span id="loading-detail">Preparing</span>
            </span>
          </div>
        </div>
        <aside class="control-panel">
          <header class="panel-header">
            <div>
              <p class="eyebrow">MJLab Web Play</p>
              <h1>MuJoCo wasm ONNX</h1>
            </div>
            <span id="status-pill" class="status-pill">Loading</span>
          </header>

          <section class="control-section">
            <label class="field-label" for="env-select">Environment</label>
            <select id="env-select" class="select-input">
              ${envRegistry
                .map(
                  (env) =>
                    `<option value="${env.id}">${env.label}${env.policy ? "" : " (scene)"}</option>`,
                )
                .join("")}
            </select>
          </section>

          <section id="mode-selector-section" class="control-section">
            <div class="mode-selector">
              <button id="mode-control-button" class="mode-btn is-active" type="button">
                <i data-lucide="sliders"></i><span>Control</span>
              </button>
              <button id="mode-nav-button" class="mode-btn" type="button">
                <i data-lucide="navigation"></i><span>Navigation</span>
              </button>
            </div>
          </section>


          <section id="gaussian-source-section" class="control-section gaussian-source-section" hidden>
            <div class="section-title">
              <i data-lucide="file-up"></i>
              <span>Gaussian Terrain</span>
            </div>
            <label id="gaussian-preset-label" class="field-label" for="gaussian-preset-select">Terrain Preset</label>
            <select id="gaussian-preset-select" class="select-input"></select>
            <label class="field-label" for="gaussian-fill-mode-select">Support Fill Mode</label>
            <select
              id="gaussian-fill-mode-select"
              class="select-input"
              title="How unresolved support-height holes are resolved while sampling the height field"
            >
              ${GAUSSIAN_SUPPORT_FILL_MODES.map(
                (mode) => `<option value="${mode.value}">${mode.label}</option>`,
              ).join("")}
            </select>
            <label
              class="file-picker"
              for="gaussian-file-input"
              title="Load a local .splat, .ply, .spz, or .sog Gaussian source for this preset"
            >
              <i data-lucide="folder-open"></i>
              <span>Load Custom Gaussian</span>
              <input
                id="gaussian-file-input"
                type="file"
                accept=".splat,.ply,.spz,.sog"
              />
            </label>
            <button
              id="gaussian-clear-button"
              class="command-button full-width"
              type="button"
              title="Remove the loaded custom Gaussian and return this preset to its bundled source"
            >
              <i data-lucide="eraser"></i><span>Clear Custom Gaussian</span>
            </button>
            <label class="range-row" for="gaussian-scale-slider">
              <span>Source Scale</span>
              <output id="gaussian-scale-output" for="gaussian-scale-slider">1.00x</output>
            </label>
            <input
              id="gaussian-scale-slider"
              type="range"
              min="0.25"
              max="8"
              step="0.05"
              value="1"
            />
            <div id="gaussian-source-status" class="inline-status">Bundle source</div>
          </section>

          <section id="spawn-picker-section" class="control-section spawn-picker-section" hidden>
            <button
              id="spawn-picker-button"
              class="command-button full-width"
              type="button"
              title="Click to place, WASD move, Space/E up, Q/C down, Enter spawn, Escape cancel"
            >
              <i data-lucide="map-pin"></i><span>Pick Spawn</span>
            </button>
          </section>

          <section id="route-planner-section" class="control-section route-planner-section" hidden>
            <div class="section-title">
              <i data-lucide="navigation"></i>
              <span>Route</span>
            </div>
            <div class="planner-settings" aria-label="Path planning controls">
              <div class="subsection-title">
                <i data-lucide="sliders-horizontal"></i>
                <span>Path Planning</span>
                <span
                  class="help-icon"
                  title="Pick a goal and tune route planner limits. Slope is dz/horizontal; step height also gates local terrain roughness."
                >
                  <i data-lucide="info"></i>
                </span>
              </div>
              <button id="global-map-button" class="command-button full-width" type="button">
                <i data-lucide="map"></i><span>Build Global Map</span>
              </button>
              <div class="route-button-row">
                <button
                  id="goal-picker-button"
                  class="command-button"
                  type="button"
                  title="Click terrain to preview; WASD move, Enter or button to finish, Escape cancel"
                >
                  <i data-lucide="flag"></i><span>Pick Goal</span>
                </button>
                <button id="route-follow-button" class="icon-button" type="button" title="Follow route">
                  <i data-lucide="navigation-2"></i>
                </button>
                <button id="route-clear-button" class="icon-button" type="button" title="Clear route">
                  <i data-lucide="x"></i>
                </button>
              </div>
              <div class="planner-limits" aria-label="Traversability limits">
                <label class="range-row" for="route-max-slope-slider">
                  <span>Passable Slope</span>
                  <output id="route-max-slope-output" for="route-max-slope-slider">55% / 29deg</output>
                </label>
                <input
                  id="route-max-slope-slider"
                  type="range"
                  min="0.2"
                  max="1.5"
                  step="0.05"
                  value="0.55"
                />
                <label class="range-row" for="route-max-step-height-slider">
                  <span>Max Step Height</span>
                  <output id="route-max-step-height-output" for="route-max-step-height-slider">0.22m</output>
                </label>
                <input
                  id="route-max-step-height-slider"
                  type="range"
                  min="0.05"
                  max="2"
                  step="0.01"
                  value="0.22"
                />
              </div>
              <div id="route-planner-status" class="inline-status"></div>
            </div>
          </section>

          <section class="button-row" aria-label="Simulation controls">
            <button
              id="pause-button"
              class="icon-button"
              type="button"
              title="Pause simulation"
              aria-label="Pause simulation"
            >
              <i data-lucide="pause"></i>
            </button>
            <button
              id="reset-button"
              class="icon-button"
              type="button"
              title="Reset simulation to the initial state"
              aria-label="Reset simulation to the initial state"
            >
              <i data-lucide="rotate-ccw"></i>
            </button>
            <button
              id="contact-button"
              class="icon-button"
              type="button"
              title="Show contact points"
              aria-label="Show contact points"
            >
              <i data-lucide="footprints"></i>
            </button>
            <button
              id="mesh-button"
              class="icon-button"
              type="button"
              title="Hide rendered meshes"
              aria-label="Hide rendered meshes"
            >
              <i data-lucide="box"></i>
            </button>
            <button
              id="heightfield-button"
              class="icon-button is-active"
              type="button"
              title="Hide heightfield mesh"
              aria-label="Hide heightfield mesh"
              hidden
            >
              <i data-lucide="mountain"></i>
            </button>
            <button
              id="skeleton-button"
              class="icon-button is-active"
              type="button"
              title="Hide robot skeleton"
              aria-label="Hide robot skeleton"
            >
              <i data-lucide="git-branch"></i>
            </button>
          </section>

          <section id="manual-command-section" class="control-section">
            <manual-command-controls id="command-controls"></manual-command-controls>
          </section>

          <section class="control-section">
            <div class="section-title">
              <i data-lucide="activity"></i>
              <span>Runtime</span>
            </div>
            <label class="range-row" for="speed-slider">
              <span>Speed</span>
              <output id="speed-output">1.00x</output>
            </label>
            <input id="speed-slider" type="range" min="0.1" max="2" step="0.05" value="1" />
            <div class="stats-grid">
              <span>Time</span><strong id="sim-time">0.00s</strong>
              <span>FPS</span><strong id="fps">0</strong>
              <span>Action</span><strong id="action-norm">0.00</strong>
              <span>ONNX</span><strong id="inference-ms">0.0ms</strong>
            </div>
          </section>

          <section class="meta-block">
            <div id="task-id"></div>
            <div id="onnx-io"></div>
          </section>
        </aside>
      </div>
    `;
    createIcons({ icons });
  }

  private bindUi(): void {
    this.requiredElement<HTMLElement>("#viewport").addEventListener("mujoco-picker-click", (event) => {
      if (!this.goalPickerActive) {
        return;
      }
      const position = (event as CustomEvent<{ position?: [number, number, number] }>).detail?.position;
      if (!position) {
        return;
      }
      void this.planRouteToGoal(position);
    });

    this.requiredElement<HTMLSelectElement>("#env-select").addEventListener("change", (event) => {
      const target = event.currentTarget as HTMLSelectElement;
      void this.loadEnvironment(target.value);
    });

    this.requiredElement<HTMLButtonElement>("#mode-control-button").addEventListener("click", () => {
      this.setMode("control");
    });
    this.requiredElement<HTMLButtonElement>("#mode-nav-button").addEventListener("click", () => {
      this.setMode("nav");
    });

    this.requiredElement<HTMLButtonElement>("#pause-button").addEventListener("click", () => {
      this.state.paused = !this.state.paused;
      this.setStatus(this.runtimeStatus());
      this.syncPauseButton();
    });
    this.requiredElement<HTMLButtonElement>("#reset-button").addEventListener("click", () => this.reset());
    this.requiredElement<HTMLButtonElement>("#contact-button").addEventListener("click", () => {
      this.state.contacts = !this.state.contacts;
      this.state.renderer?.setContactVisualization(this.state.contacts);
      this.syncVisualizationButtons();
    });
    this.requiredElement<HTMLButtonElement>("#mesh-button").addEventListener("click", () => {
      this.state.meshes = !this.state.meshes;
      this.state.renderer?.setMeshVisualization(this.state.meshes);
      this.syncVisualizationButtons();
    });
    this.requiredElement<HTMLButtonElement>("#heightfield-button").addEventListener("click", () => {
      this.state.heightfield = !this.state.heightfield;
      this.state.renderer?.setHeightfieldVisualization(this.state.heightfield);
      this.syncVisualizationButtons();
    });
    this.requiredElement<HTMLButtonElement>("#skeleton-button").addEventListener("click", () => {
      this.state.skeleton = !this.state.skeleton;
      this.state.renderer?.setSkeletonVisualization(this.state.skeleton);
      this.syncVisualizationButtons();
    });
    this.requiredElement<HTMLButtonElement>("#spawn-picker-button").addEventListener("click", () => {
      if (this.spawnPickerActive) {
        this.confirmSpawnPicker();
      } else {
        this.startSpawnPicker();
      }
    });
    this.requiredElement<HTMLButtonElement>("#goal-picker-button").addEventListener("click", () => {
      if (this.goalPickerActive) {
        this.confirmGoalPicker();
      } else {
        this.startGoalPicker();
      }
    });
    this.requiredElement<HTMLButtonElement>("#route-follow-button").addEventListener("click", () => {
      this.toggleRouteFollower();
    });
    this.requiredElement<HTMLButtonElement>("#route-clear-button").addEventListener("click", () => {
      this.clearRoutePlan();
    });
    this.requiredElement<HTMLButtonElement>("#global-map-button").addEventListener("click", () => {
      void this.buildGlobalRouteMap();
    });
    const routeMaxSlopeInput = this.requiredElement<HTMLInputElement>("#route-max-slope-slider");
    routeMaxSlopeInput.addEventListener("input", () => {
      this.updateRouteMaxSlope(Number(routeMaxSlopeInput.value), true);
    });
    routeMaxSlopeInput.addEventListener("change", () => {
      this.updateRouteMaxSlope(Number(routeMaxSlopeInput.value), false);
    });
    const routeMaxStepHeightInput = this.requiredElement<HTMLInputElement>(
      "#route-max-step-height-slider",
    );
    routeMaxStepHeightInput.addEventListener("input", () => {
      this.updateRouteMaxStepHeight(Number(routeMaxStepHeightInput.value), true);
    });
    routeMaxStepHeightInput.addEventListener("change", () => {
      this.updateRouteMaxStepHeight(Number(routeMaxStepHeightInput.value), false);
    });
    this.requiredElement<HTMLSelectElement>("#gaussian-preset-select").addEventListener("change", (event) => {
      void this.applyGaussianPresetFromInput(event.currentTarget as HTMLSelectElement);
    });
    this.requiredElement<HTMLSelectElement>("#gaussian-fill-mode-select").addEventListener("change", (event) => {
      void this.applyGaussianSupportFillModeFromInput(event.currentTarget as HTMLSelectElement);
    });
    this.requiredElement<HTMLInputElement>("#gaussian-file-input").addEventListener("change", (event) => {
      void this.loadGaussianSourceFromInput(event.currentTarget as HTMLInputElement);
    });
    this.requiredElement<HTMLButtonElement>("#gaussian-clear-button").addEventListener("click", () => {
      this.clearGaussianSourceForCurrentEnv();
    });
    const gaussianScaleInput = this.requiredElement<HTMLInputElement>("#gaussian-scale-slider");
    gaussianScaleInput.addEventListener("input", () => {
      this.updateGaussianScaleOutput(this.parseGaussianScale(gaussianScaleInput.value));
    });
    gaussianScaleInput.addEventListener("change", () => {
      void this.applyGaussianScaleFromInput(gaussianScaleInput);
    });
    const cmdControls = this.requiredElement<ManualCommandControls>("#command-controls");
    cmdControls.addEventListener("command-change", (event: any) => {
      const { key, value } = event.detail;
      this.state.policy?.setCommand({ [key]: value });
      if (this.state.policy) {
        this.syncCommandInputs(this.state.policy.command);
      }
    });

    cmdControls.addEventListener("direction-press", (event: any) => {
      const { direction, active } = event.detail;
      const env = this.state.env;
      if (!env || !env.policy) return;

      const limits = env.policy.commandLimits;
      if (active) {
        if (direction === "up") {
          const max = limits.linVelX?.[1] ?? 0.55;
          this.state.policy?.setCommand({ linVelX: Math.min(0.4, max) });
        } else if (direction === "down") {
          const min = limits.linVelX?.[0] ?? -0.55;
          this.state.policy?.setCommand({ linVelX: Math.max(-0.4, min) });
        } else if (direction === "left") {
          const max = limits.angVelZ?.[1] ?? 0.8;
          this.state.policy?.setCommand({ angVelZ: Math.min(0.6, max) });
        } else if (direction === "right") {
          const min = limits.angVelZ?.[0] ?? -0.8;
          this.state.policy?.setCommand({ angVelZ: Math.max(-0.6, min) });
        }
      } else {
        if (direction === "up" || direction === "down") {
          this.state.policy?.setCommand({ linVelX: 0 });
        } else if (direction === "left" || direction === "right") {
          this.state.policy?.setCommand({ angVelZ: 0 });
        }
      }
      
      if (this.state.policy) {
        this.syncCommandInputs(this.state.policy.command);
      }
    });

    this.requiredElement<HTMLInputElement>("#speed-slider").addEventListener("input", (event) => {
      const target = event.currentTarget as HTMLInputElement;
      this.state.speed = Number(target.value);
      this.requiredElement<HTMLOutputElement>("#speed-output").value =
        `${this.state.speed.toFixed(2)}x`;
    });

    window.addEventListener("keydown", (event) => {
      if (this.spawnPickerActive && this.handleSpawnPickerKeyDown(event)) {
        return;
      }
      if (this.goalPickerActive && this.handleGoalPickerKeyDown(event)) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        this.requiredElement<HTMLButtonElement>("#pause-button").click();
      } else if (event.code === "Backspace") {
        event.preventDefault();
        this.reset();
      }
    });
    window.addEventListener("keyup", (event) => {
      if ((this.spawnPickerActive || this.goalPickerActive) && isSpawnPickerMovementCode(event.code)) {
        event.preventDefault();
        this.pressedKeys.delete(event.code);
      }
    });
    window.addEventListener("blur", () => this.pressedKeys.clear());
  }
  private syncCommandInputs(command: CommandState | null): void {
    const env = this.controlsEnv ?? this.state.env;
    const cmdControls = this.root.querySelector<ManualCommandControls>("#command-controls");
    if (cmdControls) {
      cmdControls.env = env;
      cmdControls.command = command;
    }
  }

  private syncGaussianSourceControls(env: EnvDefinition | null): void {
    const section = this.requiredElement<HTMLElement>("#gaussian-source-section");
    const gaussianHeightfield = this.gaussianHeightfieldForEnv(env);
    section.hidden = !gaussianHeightfield;
    this.syncGaussianPresetControl(env);
    this.syncGaussianSupportFillModeControl(env);
    this.syncGaussianScaleControl(env);
    if (!gaussianHeightfield || !env) {
      this.setGaussianSourceStatus("");
      this.updateGaussianSourceControls();
      return;
    }

    const gaussianStateKey = this.gaussianStateKey(env);
    const source = this.gaussianSources.get(gaussianStateKey);
    const activeSource = this.activeGaussianSourceLabels.get(gaussianStateKey);
    this.setGaussianSourceStatus(
      source
        ? `Custom Gaussian: ${source.name} / ${formatBytes(source.bytes.byteLength)}`
        : activeSource
          ? `Bundle source: ${activeSource}`
          : "Bundle source",
    );
    this.requiredElement<HTMLInputElement>("#gaussian-file-input").value = "";
    this.updateGaussianSourceControls();
  }

  private syncGaussianPresetControl(env: EnvDefinition | null): void {
    const label = this.requiredElement<HTMLLabelElement>("#gaussian-preset-label");
    const select = this.requiredElement<HTMLSelectElement>("#gaussian-preset-select");
    const presets = this.gaussianPresetsForEnv(env);
    const hasPresets = presets.length > 0;
    label.hidden = !hasPresets;
    select.hidden = !hasPresets;
    select.replaceChildren(
      ...presets.map((preset) => {
        const option = document.createElement("option");
        option.value = preset.id;
        option.textContent = preset.label;
        return option;
      }),
    );
    const selectedPreset = this.gaussianPresetForEnv(env);
    select.value = selectedPreset?.id ?? "";
  }

  private updateGaussianSourceControls(): void {
    const section = this.requiredElement<HTMLElement>("#gaussian-source-section");
    const disabled = this.state.loading || Boolean(section.hidden);
    const env = this.controlsEnv ?? this.state.env;
    const gaussianHeightfield = this.gaussianHeightfieldForEnv(env);
    const hasSource = Boolean(
      gaussianHeightfield && env && this.gaussianSources.has(this.gaussianStateKey(env)),
    );
    const presets = this.gaussianPresetsForEnv(env);
    this.requiredElement<HTMLSelectElement>("#gaussian-preset-select").disabled =
      disabled || presets.length < 2;
    this.requiredElement<HTMLSelectElement>("#gaussian-fill-mode-select").disabled =
      disabled || !gaussianHeightfield;
    this.requiredElement<HTMLInputElement>("#gaussian-file-input").disabled = disabled;
    this.requiredElement<HTMLButtonElement>("#gaussian-clear-button").disabled =
      disabled || !hasSource;
    this.requiredElement<HTMLInputElement>("#gaussian-scale-slider").disabled = disabled;
  }

  private syncGaussianScaleControl(env: EnvDefinition | null): void {
    const scale = env && this.gaussianHeightfieldForEnv(env) ? this.gaussianScaleForEnv(env) : 1;
    const input = this.requiredElement<HTMLInputElement>("#gaussian-scale-slider");
    input.value = String(scale);
    this.updateGaussianScaleOutput(scale);
  }

  private async applyGaussianPresetFromInput(input: HTMLSelectElement): Promise<void> {
    const env = this.controlsEnv ?? this.state.env;
    if (!env || this.state.loading) {
      return;
    }
    const presets = this.gaussianPresetsForEnv(env);
    if (!presets.some((preset) => preset.id === input.value)) {
      return;
    }
    this.gaussianPresetIds.set(env.id, input.value);
    await this.loadEnvironment(env.id);
  }

  private async applyGaussianScaleFromInput(input: HTMLInputElement): Promise<void> {
    const env = this.state.env;
    if (!env || !this.gaussianHeightfieldForEnv(env) || this.state.loading) {
      return;
    }
    const scale = this.parseGaussianScale(input.value);
    this.gaussianScaleMultipliers.set(this.gaussianStateKey(env), scale);
    this.updateGaussianScaleOutput(scale);
    await this.loadEnvironment(env.id);
  }

  private syncGaussianSupportFillModeControl(env: EnvDefinition | null): void {
    const input = this.requiredElement<HTMLSelectElement>("#gaussian-fill-mode-select");
    input.value = env && this.gaussianHeightfieldForEnv(env)
      ? this.gaussianSupportFillModeForEnv(env)
      : "fallback";
  }

  private async applyGaussianSupportFillModeFromInput(input: HTMLSelectElement): Promise<void> {
    const env = this.state.env;
    if (!env || !this.gaussianHeightfieldForEnv(env) || this.state.loading) {
      return;
    }
    const mode = parseGaussianSupportFillMode(input.value);
    if (!mode) {
      this.syncGaussianSupportFillModeControl(env);
      return;
    }
    this.gaussianSupportFillModes.set(this.gaussianStateKey(env), mode);
    await this.loadEnvironment(env.id);
  }

  private updateGaussianScaleOutput(scale: number): void {
    this.requiredElement<HTMLOutputElement>("#gaussian-scale-output").value =
      `${scale.toFixed(2)}x`;
  }

  private parseGaussianScale(value: string): number {
    return clampNumber(Number(value), 0.25, 8);
  }

  private gaussianScaleForEnv(env: EnvDefinition): number {
    return this.gaussianScaleMultipliers.get(this.gaussianStateKey(env)) ??
      this.gaussianHeightfieldForEnv(env)?.sourceScaleMultiplier ??
      1;
  }

  private gaussianSupportFillModeForEnv(env: EnvDefinition): GaussianSupportFillMode {
    return this.gaussianSupportFillModes.get(this.gaussianStateKey(env)) ??
      this.gaussianHeightfieldForEnv(env)?.supportFillMode ??
      "fallback";
  }

  private envWithGaussianOverrides(env: EnvDefinition): EnvDefinition {
    const preset = this.gaussianPresetForEnv(env);
    const heightfield = this.gaussianHeightfieldForEnv(env);
    if (!heightfield) {
      return env;
    }
    return {
      ...env,
      taskId: preset?.taskId ?? env.taskId,
      heightfield: {
        ...heightfield,
        sourceScaleMultiplier: this.gaussianScaleForEnv(env),
        supportFillMode: this.gaussianSupportFillModeForEnv(env),
      },
      viewer: preset?.viewer ?? env.viewer,
    };
  }

  private gaussianPresetsForEnv(env: EnvDefinition | null): GaussianSplatPresetDefinition[] {
    return env?.gaussianPresets ?? [];
  }

  private gaussianPresetForEnv(env: EnvDefinition | null): GaussianSplatPresetDefinition | null {
    const presets = this.gaussianPresetsForEnv(env);
    if (presets.length === 0 || !env) {
      return null;
    }
    const selectedId = this.gaussianPresetIds.get(env.id);
    return presets.find((preset) => preset.id === selectedId) ?? presets[0];
  }

  private gaussianHeightfieldForEnv(env: EnvDefinition | null): GaussianSplatHeightfieldConfig | null {
    const preset = this.gaussianPresetForEnv(env);
    if (preset) {
      return preset.heightfield;
    }
    return env?.heightfield?.kind === "gaussian-splat" ? env.heightfield : null;
  }

  private gaussianStateKey(env: EnvDefinition): string {
    const preset = this.gaussianPresetForEnv(env);
    return preset ? `${env.id}:${preset.id}` : env.id;
  }

  private async loadGaussianSourceFromInput(input: HTMLInputElement): Promise<void> {
    const env = this.state.env;
    const file = input.files?.[0] ?? null;
    input.value = "";
    if (!file || !env || !this.gaussianHeightfieldForEnv(env)) {
      return;
    }
    if (!isSupportedGaussianSourceName(file.name)) {
      this.setGaussianSourceStatus(`Unsupported: ${file.name}`);
      return;
    }

    this.setGaussianSourceStatus(`Reading ${file.name}`);
    try {
      const source: GaussianHeightfieldSource = {
        name: file.name,
        bytes: new Uint8Array(await file.arrayBuffer()),
      };
      this.gaussianSources.set(this.gaussianStateKey(env), source);
      this.syncGaussianSourceControls(env);
      await this.loadEnvironment(env.id);
    } catch (error) {
      this.gaussianSources.delete(this.gaussianStateKey(env));
      this.syncGaussianSourceControls(env);
      this.setGaussianSourceStatus(error instanceof Error ? error.message : String(error));
    }
  }

  private clearGaussianSourceForCurrentEnv(): void {
    const env = this.state.env;
    if (!env || !this.gaussianHeightfieldForEnv(env)) {
      return;
    }
    this.gaussianSources.delete(this.gaussianStateKey(env));
    this.syncGaussianSourceControls(env);
    void this.loadEnvironment(env.id).catch((error) => {
      this.setGaussianSourceStatus(error instanceof Error ? error.message : String(error));
    });
  }

  private setGaussianSourceStatus(status: string): void {
    this.requiredElement<HTMLElement>("#gaussian-source-status").textContent = status;
  }

  private syncSpawnPickerControls(env: EnvDefinition | null): void {
    const section = this.requiredElement<HTMLElement>("#spawn-picker-section");
    const available = this.canUseSpawnPicker(env);
    section.hidden = !available;

    const button = this.requiredElement<HTMLButtonElement>("#spawn-picker-button");
    button.disabled = this.state.loading || !available || !this.state.renderer;
    button.innerHTML = this.spawnPickerActive
      ? `<i data-lucide="locate-fixed"></i><span>Spawn Here</span>`
      : `<i data-lucide="map-pin"></i><span>Pick Spawn</span>`;
    createIcons({ icons, root: button });
  }

  private syncRoutePlannerControls(env: EnvDefinition | null): void {
    const section = this.requiredElement<HTMLElement>("#route-planner-section");
    const available = this.canUseRoutePlanner(env);
    section.hidden = !available || this.activeMode !== "nav";
    const terrain = this.state.dynamicGaussianTerrain;
    const canBuildGlobalMap = Boolean(available && terrain && this.state.renderer);

    const globalMapButton = this.requiredElement<HTMLButtonElement>("#global-map-button");
    globalMapButton.disabled = this.state.loading || !canBuildGlobalMap || Boolean(terrain?.globalMapPending);
    globalMapButton.innerHTML = terrain?.globalMap
      ? `<i data-lucide="map-check"></i><span>Global Map Ready</span>`
      : terrain?.globalMapPending
        ? `<i data-lucide="loader"></i><span>Building Map</span>`
        : `<i data-lucide="map"></i><span>Build Global Map</span>`;

    const maxSlopeInput = this.requiredElement<HTMLInputElement>("#route-max-slope-slider");
    maxSlopeInput.disabled = this.state.loading || !available;
    maxSlopeInput.value = String(this.routePlannerMaxSlope);
    this.updateRouteMaxSlopeOutput();

    const maxStepHeightInput = this.requiredElement<HTMLInputElement>("#route-max-step-height-slider");
    maxStepHeightInput.disabled = this.state.loading || !available;
    maxStepHeightInput.value = String(this.routePlannerMaxStepHeight);
    this.updateRouteMaxStepHeightOutput();

    const goalButton = this.requiredElement<HTMLButtonElement>("#goal-picker-button");
    goalButton.disabled = this.state.loading || !available || !this.state.renderer;
    goalButton.classList.toggle("is-active", this.goalPickerActive);
    goalButton.innerHTML = this.goalPickerActive
      ? `<i data-lucide="check"></i><span>Done</span>`
      : `<i data-lucide="flag"></i><span>Pick Goal</span>`;

    const followButton = this.requiredElement<HTMLButtonElement>("#route-follow-button");
    followButton.disabled = this.state.loading || !available || !this.routePlan || !this.state.policy;
    followButton.classList.toggle("is-active", this.routeFollowEnabled);
    followButton.innerHTML = this.routeFollowEnabled
      ? `<i data-lucide="pause"></i>`
      : `<i data-lucide="navigation-2"></i>`;

    const clearButton = this.requiredElement<HTMLButtonElement>("#route-clear-button");
    clearButton.disabled = this.state.loading || (!this.routePlan && !this.routeGoalActive);
    createIcons({ icons, root: section });
  }

  private syncAllControls(env: EnvDefinition | null): void {
    const canNav = this.canUseRoutePlanner(env);
    const modeSelectorSection = this.requiredElement<HTMLElement>("#mode-selector-section");
    modeSelectorSection.hidden = !canNav;

    this.syncGaussianSourceControls(env);
    this.syncSpawnPickerControls(env);
    this.syncRoutePlannerControls(env);
    this.syncManualCommandSection(env);
  }

  private syncManualCommandSection(env: EnvDefinition | null): void {
    const manualSection = this.requiredElement<HTMLElement>("#manual-command-section");
    const hasPolicy = Boolean(env?.policy);
    manualSection.hidden = !hasPolicy || this.activeMode !== "control";

    const cmdControls = this.root.querySelector<ManualCommandControls>("#command-controls");
    if (cmdControls) {
      cmdControls.env = env;
    }
  }

  private setMode(mode: "control" | "nav"): void {
    this.activeMode = mode;
    const ctrlBtn = this.requiredElement<HTMLButtonElement>("#mode-control-button");
    const navBtn = this.requiredElement<HTMLButtonElement>("#mode-nav-button");
    ctrlBtn.classList.toggle("is-active", mode === "control");
    navBtn.classList.toggle("is-active", mode === "nav");

    if (mode === "nav") {
      this.state.policy?.zeroCommand();
      if (this.state.policy) {
        this.syncCommandInputs(this.state.policy.command);
      }
    } else {
      this.routeFollowEnabled = false;
      this.state.policy?.zeroCommand();
      if (this.state.policy) {
        this.syncCommandInputs(this.state.policy.command);
      }
      this.setRoutePlannerStatus("");
      this.setStatus(this.runtimeStatus());
    }

    const env = this.controlsEnv ?? this.state.env;
    this.syncAllControls(env);
  }

  private updateRouteMaxSlope(value: number, previewOnly: boolean): void {
    this.routePlannerMaxSlope = clampNumber(value, 0.2, 1.5);
    this.updateRouteMaxSlopeOutput();
    if (previewOnly) {
      this.scheduleRoutePlannerReplan();
      return;
    }
    this.cancelRoutePlannerReplan();
    this.replanActiveRouteGoal();
  }

  private updateRouteMaxSlopeOutput(): void {
    const percent = Math.round(this.routePlannerMaxSlope * 100);
    const degrees = Math.round((Math.atan(this.routePlannerMaxSlope) * 180) / Math.PI);
    this.requiredElement<HTMLOutputElement>("#route-max-slope-output").value =
      `${percent}% / ${degrees}deg`;
  }

  private updateRouteMaxStepHeight(value: number, previewOnly: boolean): void {
    this.routePlannerMaxStepHeight = clampNumber(value, 0.05, 2);
    this.updateRouteMaxStepHeightOutput();
    if (previewOnly) {
      this.scheduleRoutePlannerReplan();
      return;
    }
    this.cancelRoutePlannerReplan();
    this.replanActiveRouteGoal();
  }

  private updateRouteMaxStepHeightOutput(): void {
    this.requiredElement<HTMLOutputElement>("#route-max-step-height-output").value =
      `${this.routePlannerMaxStepHeight.toFixed(2)}m`;
  }

  private routePlannerLimitsLabel(): string {
    return `slope ${Math.round(this.routePlannerMaxSlope * 100)}%, step ${this.routePlannerMaxStepHeight.toFixed(2)}m`;
  }

  private scheduleRoutePlannerReplan(): void {
    this.cancelRoutePlannerReplan();
    if (!this.routeGoal) {
      return;
    }
    this.routePlannerReplanTimer = window.setTimeout(() => {
      this.routePlannerReplanTimer = null;
      this.replanActiveRouteGoal();
    }, 300);
  }

  private cancelRoutePlannerReplan(): void {
    if (this.routePlannerReplanTimer === null) {
      return;
    }
    window.clearTimeout(this.routePlannerReplanTimer);
    this.routePlannerReplanTimer = null;
  }

  private replanActiveRouteGoal(): void {
    if (!this.routeGoal) {
      return;
    }
    void this.planRouteToGoal(this.routeGoal);
  }

  private setRoutePlannerStatus(status: string, kind: RoutePlannerStatusKind = "idle"): void {
    const element = this.requiredElement<HTMLElement>("#route-planner-status");
    element.textContent = status;
    const activeKind = status ? kind : "idle";
    element.classList.toggle("is-busy", activeKind === "busy");
    element.classList.toggle("is-success", activeKind === "success");
    element.classList.toggle("is-error", activeKind === "error");
  }

  private syncPauseButton(): void {
    const button = this.requiredElement<HTMLButtonElement>("#pause-button");
    button.innerHTML = this.state.paused
      ? `<i data-lucide="play"></i>`
      : `<i data-lucide="pause"></i>`;
    this.setButtonTooltip(button, this.state.paused ? "Resume simulation" : "Pause simulation");
    createIcons({ icons, root: button });
  }

  private syncVisualizationButtons(): void {
    this.setButtonTooltip(
      this.requiredElement<HTMLButtonElement>("#reset-button"),
      "Reset simulation to the initial state",
    );

    const contactButton = this.requiredElement<HTMLButtonElement>("#contact-button");
    contactButton.classList.toggle("is-active", this.state.contacts);
    this.setButtonTooltip(
      contactButton,
      this.state.contacts ? "Hide contact points" : "Show contact points",
    );

    const meshButton = this.requiredElement<HTMLButtonElement>("#mesh-button");
    meshButton.classList.toggle("is-active", this.state.meshes);
    this.setButtonTooltip(meshButton, this.state.meshes ? "Hide rendered meshes" : "Show rendered meshes");

    const hasGaussian = Boolean(this.state.env?.heightfield);

    const heightfieldButton = this.requiredElement<HTMLButtonElement>("#heightfield-button");
    heightfieldButton.hidden = !hasGaussian;
    heightfieldButton.classList.toggle("is-active", this.state.heightfield);
    this.setButtonTooltip(
      heightfieldButton,
      this.state.heightfield ? "Hide heightfield mesh" : "Show heightfield mesh",
    );

    const skeletonButton = this.requiredElement<HTMLButtonElement>("#skeleton-button");
    skeletonButton.classList.toggle("is-active", this.state.skeleton);
    this.setButtonTooltip(
      skeletonButton,
      this.state.skeleton ? "Hide robot skeleton" : "Show robot skeleton",
    );

  }

  private setButtonTooltip(button: HTMLButtonElement, label: string): void {
    button.title = label;
    button.setAttribute("aria-label", label);
  }

  private updateStats(time: number): void {
    this.state.frameCount += 1;
    if (time - this.state.lastFpsAt >= 500) {
      this.state.fps = (1000 * this.state.frameCount) / (time - this.state.lastFpsAt);
      this.state.frameCount = 0;
      this.state.lastFpsAt = time;
    }

    const simTime = this.state.data?.time ?? 0;
    this.requiredElement<HTMLElement>("#sim-time").textContent = `${simTime.toFixed(2)}s`;
    this.requiredElement<HTMLElement>("#fps").textContent = String(Math.round(this.state.fps));
    this.requiredElement<HTMLElement>("#action-norm").textContent =
      this.state.policy ? (this.state.policyStats?.actionNorm.toFixed(2) ?? "0.00") : "n/a";
    this.requiredElement<HTMLElement>("#inference-ms").textContent =
      this.state.policy ? `${(this.state.policyStats?.lastInferenceMs ?? 0).toFixed(1)}ms` : "n/a";
    this.requiredElement<HTMLElement>("#task-id").textContent = this.state.env?.taskId ?? "";
    this.requiredElement<HTMLElement>("#onnx-io").textContent =
      this.state.policyStats
        ? `${this.state.policyStats.inputName} -> ${this.state.policyStats.outputName}`
        : this.state.env && !this.state.env.policy
          ? "ONNX policy pending"
          : "";
  }

  private updateControlsEnabled(): void {
    const disabled = this.state.loading;
    for (const selector of ["button", "select", "input"]) {
      this.root.querySelectorAll<HTMLButtonElement | HTMLInputElement | HTMLSelectElement>(selector)
        .forEach((element) => {
          element.disabled = disabled;
        });
    }
    this.updatePolicyControls();
    this.updateGaussianSourceControls();
    this.syncAllControls(this.controlsEnv ?? this.state.env);
  }

  private updatePolicyControls(): void {
    const disabled = this.state.loading || !this.state.policy;
    const cmdControls = this.root.querySelector<ManualCommandControls>("#command-controls");
    if (cmdControls) {
      cmdControls.disabled = disabled;
    }
  }

  private runtimeStatus(): string {
    if (this.spawnPickerActive) {
      return "Pick Spawn";
    }
    if (this.goalPickerActive) {
      return "Pick Goal";
    }
    if (this.routeFollowEnabled) {
      return "Following";
    }
    if (!this.state.env?.policy) {
      return this.state.paused ? "Scene preview" : "Physics preview";
    }
    return this.state.paused ? "Paused" : "Running";
  }

  private setStatus(status: string): void {
    this.requiredElement<HTMLElement>("#status-pill").textContent = status;
  }

  private setLoadingState(loading: boolean, title: string, detail = ""): void {
    this.state.loading = loading;
    this.setStatus(title);
    this.root.classList.toggle("is-loading", loading);
    const overlay = this.root.querySelector<HTMLElement>("#loading-overlay");
    const titleElement = this.root.querySelector<HTMLElement>("#loading-title");
    const detailElement = this.root.querySelector<HTMLElement>("#loading-detail");
    overlay?.classList.toggle("is-visible", loading);
    if (titleElement) {
      titleElement.textContent = title;
    }
    if (detailElement) {
      detailElement.textContent = detail;
    }
  }

  private heightfieldProgressDetail(progress: HeightfieldGenerationProgress): string {
    if (progress.stage === "loading-source") {
      return progress.completed >= progress.total
        ? `Loaded ${progress.detail ?? "source"}`
        : `Loading ${progress.detail ?? "source"}`;
    }
    if (progress.stage === "support-tiles") {
      const total = Math.max(1, progress.total);
      const percent = Math.round((100 * progress.completed) / total);
      return `Generating support tiles ${progress.completed}/${progress.total} (${percent}%)`;
    }
    const total = Math.max(1, progress.total);
    const percent = Math.round((100 * progress.completed) / total);
    return `Sampling MuJoCo height field ${progress.completed}/${progress.total} rows (${percent}%)`;
  }

  private requiredElement<T extends Element>(selector: string): T {
    const element = this.root.querySelector<T>(selector);
    if (!element) {
      throw new Error(`Missing element: ${selector}`);
    }
    return element;
  }
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.text();
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function yieldForPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve());
    });
  });
}

function heightfieldWindowToPlannerGrid(window: GaussianHeightfieldWindow): HeightfieldPathGrid {
  return {
    nrow: window.config.nrow,
    ncol: window.config.ncol,
    adr: 0,
    sizeX: window.config.sizeX,
    sizeY: window.config.sizeY,
    sizeZ: 1,
    originX: window.centerX,
    originY: window.centerY,
    cellSizeX: (2 * window.config.sizeX) / (window.config.ncol - 1),
    cellSizeY: (2 * window.config.sizeY) / (window.config.nrow - 1),
    data: window.heights,
  };
}

function optionsForPlanningWindow(
  options: HeightfieldPathOptions,
  window: GaussianHeightfieldWindow,
): HeightfieldPathOptions {
  return {
    ...options,
    maxExpanded: Math.max(
      options.maxExpanded ?? 0,
      window.config.nrow * window.config.ncol,
    ),
  };
}

function terrainWindowBounds(
  centerX: number,
  centerY: number,
  sizeX: number,
  sizeY: number,
): { minX: number; maxX: number; minY: number; maxY: number } {
  return {
    minX: centerX - sizeX,
    maxX: centerX + sizeX,
    minY: centerY - sizeY,
    maxY: centerY + sizeY,
  };
}

function pointInsideBounds(
  point: readonly [number, number],
  bounds: { minX: number; maxX: number; minY: number; maxY: number },
): boolean {
  return point[0] >= bounds.minX &&
    point[0] <= bounds.maxX &&
    point[1] >= bounds.minY &&
    point[1] <= bounds.maxY;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function isSupportedGaussianSourceName(fileName: string): boolean {
  return /\.(splat|ply|spz|sog)$/i.test(fileName);
}

function parseGaussianSupportFillMode(value: string): GaussianSupportFillMode | null {
  return GAUSSIAN_SUPPORT_FILL_MODES.some((mode) => mode.value === value)
    ? value as GaussianSupportFillMode
    : null;
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function routeFollowerCommand(
  env: EnvDefinition,
  robotX: number,
  robotY: number,
  robotYaw: number,
  targetX: number,
  targetY: number,
  scale: number,
): CommandState {
  if (env.policy?.routeFollowerMode === "holonomic") {
    return holonomicRouteFollowerCommand(
      env.policy.commandLimits,
      robotX,
      robotY,
      robotYaw,
      targetX,
      targetY,
      scale,
    );
  }
  return headingAlignedRouteFollowerCommand(env, robotX, robotY, robotYaw, targetX, targetY, scale);
}

function headingAlignedRouteFollowerCommand(
  env: EnvDefinition,
  robotX: number,
  robotY: number,
  robotYaw: number,
  targetX: number,
  targetY: number,
  scale: number,
): CommandState {
  const dx = targetX - robotX;
  const dy = targetY - robotY;
  const distance = Math.max(1e-6, Math.hypot(dx, dy));
  const desiredYaw = Math.atan2(dy, dx);
  const yawError = wrapPi(desiredYaw - robotYaw);
  const localX = Math.cos(robotYaw) * dx + Math.sin(robotYaw) * dy;
  const localY = -Math.sin(robotYaw) * dx + Math.cos(robotYaw) * dy;
  const turnScale = clampNumber(1 - Math.min(0.75, Math.abs(yawError) / Math.PI), 0.25, 1);
  const speed = clampNumber((distance / scale) * 0.65, 0.18, 0.65) * turnScale;

  return clampCommand(
    {
      linVelX: (localX / distance) * speed,
      linVelY: (localY / distance) * speed * 0.75,
      angVelZ: yawError * 1.35,
    },
    env.policy?.commandLimits,
  );
}

function holonomicRouteFollowerCommand(
  limits: CommandLimits,
  robotX: number,
  robotY: number,
  robotYaw: number,
  targetX: number,
  targetY: number,
  scale: number,
): CommandState {
  const dx = targetX - robotX;
  const dy = targetY - robotY;
  const dist = Math.hypot(dx, dy);

  if (dist < 0.08 * scale) {
    return { linVelX: 0, linVelY: 0, angVelZ: 0 };
  }

  const targetYaw = Math.atan2(dy, dx);
  const yawError = wrapPi(targetYaw - robotYaw);
  const absYawError = Math.abs(yawError);
  const turnSpeedScale = absYawError > 1.25 ? 0.58 : absYawError > 0.55 ? 0.76 : 1.0;
  const baseSpeed = clampNumber((dist / scale) * 0.75, 0.22, 0.9);
  const speed = baseSpeed * turnSpeedScale;
  const worldVx = (dx / dist) * speed;
  const worldVy = (dy / dist) * speed;
  const c = Math.cos(robotYaw);
  const s = Math.sin(robotYaw);

  return clampCommand(
    {
      linVelX: c * worldVx + s * worldVy,
      linVelY: -s * worldVx + c * worldVy,
      angVelZ: absYawError < 0.08 ? 0 : yawError * 0.78,
    },
    limits,
  );
}

function clampCommand(command: CommandState, limits?: CommandLimits): CommandState {
  if (!limits) {
    return command;
  }
  return {
    linVelX: clampNumber(command.linVelX, limits.linVelX[0], limits.linVelX[1]),
    linVelY: clampNumber(command.linVelY, limits.linVelY[0], limits.linVelY[1]),
    angVelZ: clampNumber(command.angVelZ, limits.angVelZ[0], limits.angVelZ[1]),
  };
}

function distance2d(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function isSpawnPickerMovementCode(code: string): boolean {
  return (
    code === "KeyW" ||
    code === "KeyA" ||
    code === "KeyS" ||
    code === "KeyD" ||
    code === "KeyE" ||
    code === "KeyQ" ||
    code === "KeyC" ||
    code === "Space" ||
    code === "ShiftLeft" ||
    code === "ShiftRight"
  );
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

const root = document.querySelector<HTMLElement>("#app");
if (!root) {
  throw new Error("Missing #app root");
}

const app = new WebPlayApp(root);
app.start().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  root.innerHTML = `<div class="fatal-error">${message}</div>`;
});
