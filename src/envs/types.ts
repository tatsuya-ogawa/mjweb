export type EnvObservationKind =
  | "velocity-flat-v1"
  | "velocity-rough-v1"
  | "tracking-motion-v1";

export interface TerrainScanConfig {
  /** Body whose world-frame position+yaw is used to anchor the ray grid. */
  frameBody: string;
  /** Body id (or its parent) to exclude from raycasts. Matches `exclude_parent_body=True`. */
  excludeBody: string;
  /** Total grid extent in body-local X (forward). */
  sizeX: number;
  /** Total grid extent in body-local Y (left). */
  sizeY: number;
  /** Spacing between rays in meters. */
  resolution: number;
  /** Rays beyond this distance report a miss. */
  maxDistance: number;
  /** Geom groups (0-5) to include. mjlab defaults to (0,) for terrain. */
  geomGroups: number[];
}

export interface CommandState {
  linVelX: number;
  linVelY: number;
  angVelZ: number;
}

export interface CommandLimits {
  linVelX: [number, number];
  linVelY: [number, number];
  angVelZ: [number, number];
}

export interface EnvAsset {
  url: string;
  vfsPath: string;
}

export interface InitialFreeJointState {
  jointName: string;
  qpos: [number, number, number, number, number, number, number];
  qvel?: [number, number, number, number, number, number];
}

export interface TerrainSnapInitialState {
  jointName: string;
  excludeBody?: string;
  geomGroups: number[];
  rayStartHeight: number;
  maxDistance: number;
}

export interface BaseHeightfieldConfig {
  /** MuJoCo hfield rows and columns. Rows map to local Y, columns map to local X. */
  nrow: number;
  ncol: number;
  /** MuJoCo hfield half extents in meters. */
  sizeX: number;
  sizeY: number;
  /** MuJoCo hfield z scale and base thickness. */
  elevationScale: number;
  baseThickness: number;
  smoothingPasses: number;
  /** Local max sampling radius for the initial base height. Defaults to a single bilinear sample. */
  startHeightSampleRadius?: number;
  /** Initial X/Y selection. "support" searches dense support-height chunks for a flat supported spawn. */
  startPlacement?: "configured" | "support";
  /** Maximum number of dense support chunks considered when startPlacement is "support". */
  startSearchChunkLimit?: number;
  /** Required supported radius around the initial X/Y position in meters. */
  startSupportRadius?: number;
  /** Required supported-sample ratio inside startSupportRadius. */
  startMinSupportRatio?: number;
  /** Maximum local terrain height range inside startSupportRadius. */
  startMaxHeightRange?: number;
  startX: number;
  startY: number;
  robotBaseHeight: number;
}

export interface ProceduralSlopeHeightfieldConfig extends BaseHeightfieldConfig {
  kind: "procedural-slope";
  /** Smooth slope rise used by procedural-only slope envs. */
  slopeHeight: number;
}

export interface GaussianSourceTransformConfig {
  /** Remote source URL for this transform bundle. */
  sourceUrl?: string;
  /** Remote source URLs for this transform bundle, tried in order. */
  sourceUrls?: readonly string[];
  /** 4x4 row-major source transform, or per-file transforms keyed by basename. */
  matrix?: readonly number[] | Record<string, readonly number[]>;
  /** Source scale applied before projection. */
  scale?: number;
  /** Optional starting pose override after projection. */
  spawn?:
    | readonly [number, number]
    | readonly [number, number, number]
    | { x: number; y: number; yaw?: number };
}

export interface GaussianSplatHeightfieldConfig extends BaseHeightfieldConfig {
  kind: "gaussian-splat";
  /** Bundle name under public/envs/<env id>/splats. Supported Gaussian files become source candidates. */
  sourceBundleId?: string;
  /** Optional fallback/override source URLs. The first fetchable .sog/.spz/.splat/.ply source is projected. */
  sourceUrls?: string[];
  /** Optional transform config for sources that do not ship a sibling transform.json. */
  sourceTransformConfig?: GaussianSourceTransformConfig;
  /** Source X/Z center used for projection. Defaults to source bounds center. */
  sourceCenter?: "bounds" | "origin" | "density";
  /** Projection mode. "fit" scales the source into sizeX/sizeY; "metric" treats source units as meters. */
  sourceScaleMode?: "fit" | "metric";
  /** Meters represented by one source coordinate unit in metric mode. Defaults to 1. */
  sourceMetersPerUnit?: number;
  /** Multiplies the projected 3DGS footprint and height before sampling. */
  sourceScaleMultiplier?: number;
  /** Height used for non-metric source fitting. Defaults to elevationScale. */
  sourceFitHeight?: number;
  /** Extra meters added around metric-mode source bounds when expanding the hfield. */
  sourceBoundsPadding?: number;
  /** Optional metric-mode half extent caps. Use only for an intentionally cropped local window. */
  sourceMaxSizeX?: number;
  sourceMaxSizeY?: number;
  /** Target hfield cell size in meters after metric-mode source expansion. */
  sourceTargetCellSize?: number;
  /** Optional metric-mode hfield resolution caps. */
  sourceMaxRows?: number;
  sourceMaxCols?: number;
  /** WebGPU support-height chunk size in projected meters. Defaults to the TPS example chunk size. */
  supportChunkSize?: number;
  /** Number of cells per support-height chunk edge. Defaults to the TPS example resolution. */
  supportChunkResolution?: number;
  /** Extra binning padding around splats for support-height chunks. Defaults to the TPS example padding. */
  supportChunkPadding?: number;
  /** Density threshold used to accept a support-height voxel. Defaults to the TPS example threshold. */
  supportDensityThreshold?: number;
  /** Number of support-height hole-fill passes inside each chunk. Defaults to the TPS example value. */
  supportFillIterations?: number;
  /** Maximum local height range allowed when filling support-height holes. Defaults to the TPS example value. */
  supportFillMaxHeightRange?: number;
  /** How unresolved support-height holes are resolved at sample time. */
  supportFillMode?: "nearby" | "fallback" | "min";
  /** WebGPU density smoothing iterations before support-height extraction. Defaults to the TPS example value. */
  supportSmoothIterations?: number;
  /** Maximum number of support-height chunks generated concurrently. */
  supportMaxConcurrentJobs?: number;
  /** Keep a fixed-size local MuJoCo hfield and regenerate it around the robot while running. */
  dynamicWindow?: boolean;
  /** Distance from the current hfield center before a dynamic window refresh is requested. */
  dynamicWindowUpdateDistance?: number;
  /** Move the initial local Y position onto the projected source footprint when a source exists. */
  startYOnSourceFootprint?: boolean;
  /** Move the initial local X/Y position to the projected Gaussian origin when a source exists. */
  startAtGaussianOrigin?: boolean;
}

export type HeightfieldConfig =
  | GaussianSplatHeightfieldConfig
  | ProceduralSlopeHeightfieldConfig;

export interface GaussianSplatPresetDefinition {
  id: string;
  label: string;
  taskId?: string;
  heightfield: GaussianSplatHeightfieldConfig;
  viewer?: {
    followBody: string;
    distance: number;
    azimuthDeg: number;
    elevationDeg: number;
  };
}

export interface IdealPdControlConfig {
  stiffness: number;
  damping: number;
}

export interface PolicyDefinition {
  onnxUrl: string;
  inputSize: number;
  outputSize: number;
  defaultJointPos: number[];
  actionScale: number[];
  jointNames: string[];
  actuatorNames: string[];
  imuLinearVelocitySensor: string;
  imuAngularVelocitySensor: string;
  rootJointName: string;
  keyframeId: number;
  commandDefaults: CommandState;
  commandLimits: CommandLimits;
  controlDt: number;
  decimation: number;
  actionClip?: number;
  routeFollowerMode?: "heading-aligned" | "holonomic";
  jointControlMode?: "position" | "ideal-pd";
  idealPd?: IdealPdControlConfig;
  /** Optional terrain height scan inputs (appended to observation tail for rough envs). */
  terrainScan?: TerrainScanConfig;
}

export interface EnvDefinition {
  id: string;
  label: string;
  taskId: string;
  sceneXmlUrl: string;
  assets: EnvAsset[];
  observationKind: EnvObservationKind;
  policy?: PolicyDefinition;
  sim: {
    timestep: number;
  };
  render?: {
    mirrorMeshY?: boolean;
    visualMeshManifestUrl?: string;
  };
  heightfield?: HeightfieldConfig;
  gaussianPresets?: GaussianSplatPresetDefinition[];
  initialState?: {
    freeJoints?: InitialFreeJointState[];
    terrainSnap?: TerrainSnapInitialState;
  };
  viewer: {
    followBody: string;
    distance: number;
    azimuthDeg: number;
    elevationDeg: number;
  };
}

export interface PolicyEnvDefinition extends EnvDefinition {
  policy: PolicyDefinition;
}
