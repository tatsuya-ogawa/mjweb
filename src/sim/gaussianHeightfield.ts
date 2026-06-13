import {
  ChunkVoxelManager,
  computeBounds,
  packSplats,
  SplatSpatialBinning,
  type HeightField,
  type PackedSplatsForGpu,
} from "web-3dgs-to-pc/browser";
import type {
  BaseHeightfieldConfig,
  EnvDefinition,
  GaussianSplatHeightfieldConfig,
  HeightfieldConfig,
  ProceduralSlopeHeightfieldConfig,
} from "../envs/types";
import { gaussianSourceUrlsForBundle } from "../envs/gaussianSourceBundles";

type Vec3 = [number, number, number];

const SPLAT_VISUAL_LIFT = 0.035;
const DEFAULT_SUPPORT_CHUNK_SIZE = 8;
const DEFAULT_SUPPORT_CHUNK_RESOLUTION = 64;
const DEFAULT_SUPPORT_CHUNK_PADDING = 2;
const DEFAULT_SUPPORT_DENSITY_THRESHOLD = 120;
const DEFAULT_SUPPORT_FILL_ITERATIONS = 18;
const DEFAULT_SUPPORT_FILL_MAX_HEIGHT_RANGE = 1.2;
const DEFAULT_SUPPORT_SMOOTH_ITERATIONS = 0;
const DEFAULT_SUPPORT_MAX_CONCURRENT_JOBS = 2;
const HEIGHTFIELD_SAMPLE_PROGRESS_ROWS = 8;
const DEFAULT_START_SEARCH_CHUNK_LIMIT = 8;
const DEFAULT_START_SUPPORT_RADIUS = 0.45;
const DEFAULT_START_MIN_SUPPORT_RATIO = 0.82;
const DEFAULT_START_MAX_HEIGHT_RANGE = 0.45;

interface GaussianSplat {
  center: Vec3;
  scales: Vec3;
  rotation: [number, number, number, number];
  opacity: number;
  color: [number, number, number];
}

interface SplatBounds {
  min: Vec3;
  max: Vec3;
}

interface GaussianSplatSet {
  splats: GaussianSplat[];
  bounds?: SplatBounds;
  packed?: PackedSplatsForGpu;
}

interface PreparedHeightfield {
  config: HeightfieldConfig;
  heights: Float32Array;
  startHeight: number;
  visualSource: GaussianSplatVisualSource | null;
  windowCenterX: number;
  windowCenterY: number;
  dynamicHeightfield: DynamicGaussianHeightfield | null;
  transformConfig?: TransformConfig | null;
  customSpawn?: [number, number, number] | null;
}

export type HeightfieldGenerationStage = "loading-source" | "support-tiles" | "sampling";

export interface HeightfieldGenerationProgress {
  stage: HeightfieldGenerationStage;
  completed: number;
  total: number;
  detail?: string;
}

export interface HeightfieldPreparationOptions {
  signal?: AbortSignal;
  onProgress?: (progress: HeightfieldGenerationProgress) => void;
  userScale?: number;
}

export interface GaussianHeightfieldSource {
  name: string;
  bytes: Uint8Array;
}

export interface GaussianSplatVisualSource {
  source: GaussianHeightfieldSource;
  matrix: number[];
  boundsRadius: number;
}

export interface PreparedHeightfieldScene {
  xml: string;
  visualSource: GaussianSplatVisualSource | null;
  dynamicHeightfield: DynamicGaussianHeightfield | null;
  transformConfig?: TransformConfig | null;
}

export interface GaussianHeightfieldWindow {
  config: GaussianSplatHeightfieldConfig;
  centerX: number;
  centerY: number;
  heights: Float32Array;
}

interface GaussianSupportHeightfieldContext {
  config: GaussianSplatHeightfieldConfig;
  binning: SplatSpatialBinning;
  manager: ChunkVoxelManager;
  boundsMin: Vec3;
  boundsMax: Vec3;
  chunkSize: number;
  chunkResolution: number;
  maxConcurrentJobs: number;
}

interface SupportStartCandidate {
  x: number;
  y: number;
  score: number;
  supportRatio: number;
  heightRange: number;
  tile: [number, number];
}

export class DynamicGaussianHeightfield {
  readonly config: GaussianSplatHeightfieldConfig;
  readonly updateDistance: number;
  readonly initialCenterX: number;
  readonly initialCenterY: number;

  constructor(
    private readonly context: GaussianSupportHeightfieldContext,
    initialCenterX: number,
    initialCenterY: number,
  ) {
    this.config = context.config;
    this.initialCenterX = initialCenterX;
    this.initialCenterY = initialCenterY;
    this.updateDistance = Math.max(
      0.1,
      context.config.dynamicWindowUpdateDistance ?? 0.5 * context.chunkSize,
    );
  }

  async generateWindow(
    centerX: number,
    centerY: number,
    options: HeightfieldPreparationOptions = {},
  ): Promise<GaussianHeightfieldWindow> {
    const heights = await sampleSupportHeightfieldWindow(
      this.context,
      centerX,
      centerY,
      options,
    );
    clampHeightfield(heights, this.config.elevationScale);
    return {
      config: this.config,
      centerX,
      centerY,
      heights,
    };
  }

  async generateRoutePlanningWindow(
    start: readonly [number, number],
    goal: readonly [number, number],
    options: HeightfieldPreparationOptions = {},
  ): Promise<GaussianHeightfieldWindow> {
    const config = routePlanningWindowConfig(this.config, start, goal);
    const centerX = 0.5 * (start[0] + goal[0]);
    const centerY = 0.5 * (start[1] + goal[1]);
    const heights = await sampleSupportHeightfieldWindow(
      this.context,
      centerX,
      centerY,
      options,
      config,
    );
    clampHeightfield(heights, config.elevationScale);
    return {
      config,
      centerX,
      centerY,
      heights,
    };
  }

  async generateGlobalPlanningWindow(
    options: HeightfieldPreparationOptions = {},
  ): Promise<GaussianHeightfieldWindow> {
    const { config, centerX, centerY } = globalPlanningWindowConfig(
      this.config,
      this.context.boundsMin,
      this.context.boundsMax,
    );
    const heights = await sampleSupportHeightfieldWindow(
      this.context,
      centerX,
      centerY,
      options,
      config,
    );
    clampHeightfield(heights, config.elevationScale);
    return {
      config,
      centerX,
      centerY,
      heights,
    };
  }
}

interface SplatProjection {
  boundsMin: Vec3;
  boundsMax: Vec3;
  rangeY: number;
  horizontalScale: number;
  heightScale: number;
  lateralSign: number;
  verticalSign: number;
  centerX: number;
  centerZ: number;
  minY: number;
  maxY: number;
  maxHeight: number;
  footprintMinX: number;
  footprintMaxX: number;
  footprintMinY: number;
  footprintMaxY: number;
}

interface GaussianDebugState {
  sourceName: string;
  splatCount: number;
  boundsMin: Vec3;
  boundsMax: Vec3;
  center: [number, number];
  horizontalScale: number;
  heightScale: number;
  lateralSign: number;
  verticalSign: number;
  footprintMin: [number, number];
  footprintMax: [number, number];
  maxHeight: number;
  hfield: {
    nrow: number;
    ncol: number;
    sizeX: number;
    sizeY: number;
    elevationScale: number;
    startX: number;
    startY: number;
    startHeight: number;
  };
  matrix: number[];
  boundsRadius: number;
}

declare global {
  interface Window {
    __mjwebGaussianDebug?: GaussianDebugState;
  }
}

interface PlyHeader {
  format: "ascii" | "binary_little_endian";
  vertexCount: number;
  vertexProperties: PlyProperty[];
  headerEnd: number;
}

interface PlyProperty {
  name: string;
  type: PlyScalarType;
}

type PlyScalarType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double";

export async function prepareHeightfieldSceneXml(
  env: EnvDefinition,
  xml: string,
  source?: GaussianHeightfieldSource,
  options: HeightfieldPreparationOptions = {},
): Promise<string> {
  return (await prepareHeightfieldScene(env, xml, source, options)).xml;
}

export async function prepareHeightfieldScene(
  env: EnvDefinition,
  xml: string,
  source?: GaussianHeightfieldSource,
  options: HeightfieldPreparationOptions = {},
): Promise<PreparedHeightfieldScene> {
  if (!env.heightfield) {
    return {
      xml,
      visualSource: null,
      dynamicHeightfield: null,
    };
  }

  const prepared = await createHeightfield(env.heightfield, source, options, env.id);
  const doc = parseXml(xml);
  const hfield = doc.createElement("hfield");
  hfield.setAttribute("name", "gs_heightfield");
  hfield.setAttribute(
    "size",
    [
      prepared.config.sizeX,
      prepared.config.sizeY,
      prepared.config.elevationScale,
      prepared.config.baseThickness,
    ].map(formatNumber).join(" "),
  );
  hfield.setAttribute("nrow", String(prepared.config.nrow));
  hfield.setAttribute("ncol", String(prepared.config.ncol));
  hfield.setAttribute("elevation", heightfieldElevationString(prepared));

  const asset = ensureAssetElement(doc);
  asset.appendChild(doc.createTextNode("\n    "));
  asset.appendChild(hfield);
  asset.appendChild(doc.createTextNode("\n  "));

  replaceTerrainBody(doc, prepared);
  updateInitKeyframe(doc, prepared);

  return {
    xml: new XMLSerializer().serializeToString(doc),
    visualSource: prepared.visualSource,
    dynamicHeightfield: prepared.dynamicHeightfield,
  };
}

async function createHeightfield(
  config: HeightfieldConfig,
  source?: GaussianHeightfieldSource,
  options: HeightfieldPreparationOptions = {},
  envId?: string,
): Promise<PreparedHeightfield> {
  throwIfAborted(options.signal);
  if (config.kind === "procedural-slope") {
    return createProceduralSlopeHeightfield(config);
  }

  options.onProgress?.({
    stage: "loading-source",
    completed: 0,
    total: 1,
    detail: source ? source.name : "Bundled source",
  });
  const loaded = source
    ? bakeGaussianSource(
        await loadGaussianSource(source.name, source.bytes),
        source.name,
        defaultTransformMatrix(source.name),
      )
    : await loadFirstGaussianSource(config, envId);
  if (loaded && source) {
    loaded.source.bytes = source.bytes;
  }
  throwIfAborted(options.signal);
  options.onProgress?.({
    stage: "loading-source",
    completed: 1,
    total: 1,
    detail: loaded?.source.name ?? "No source",
  });
  if (!loaded || loaded.splats.length === 0) {
    throw new Error("Gaussian heightfield requires a non-empty splat source");
  }
  const splats = loaded.splats;

  // Resolve scale factor using options.userScale or transformConfig.scale
  let scale = 1.0;
  if (options.userScale !== undefined) {
    scale = options.userScale;
  } else if (loaded.transformConfig && typeof loaded.transformConfig.scale === "number") {
    scale = loaded.transformConfig.scale;
  } else if (config.kind === "gaussian-splat" && typeof config.sourceScaleMultiplier === "number") {
    scale = config.sourceScaleMultiplier;
  }

  let activeConfig = config;
  if (config.kind === "gaussian-splat") {
    activeConfig = {
      ...config,
      sourceScaleMultiplier: scale,
    };
  }

  const projection = computeSplatProjection(
    activeConfig,
    splats,
    loaded.bounds,
  );
  let resolvedConfig = configForProjection(activeConfig, projection);

  // Resolve custom spawn config if specified in transform.json
  let customSpawn: [number, number, number] | null = null;
  if (loaded.transformConfig && loaded.transformConfig.spawn) {
    const spawn = loaded.transformConfig.spawn;
    let sx = 0;
    let sy = 0;
    let syaw = 0;
    if (Array.isArray(spawn)) {
      sx = spawn[0] ?? 0;
      sy = spawn[1] ?? 0;
      syaw = spawn[2] ?? 0;
    } else if (typeof spawn === "object" && spawn !== null) {
      sx = spawn.x ?? 0;
      sy = spawn.y ?? 0;
      syaw = spawn.yaw ?? 0;
    }
    resolvedConfig = {
      ...resolvedConfig,
      startX: sx,
      startY: sy,
    };
    customSpawn = [sx, sy, syaw];
  }

  const context = await createGaussianSupportHeightfieldContext(
    resolvedConfig,
    packGaussianSplatsForGpu(splats),
    projection,
    options,
  );

  // If a custom spawn coordinate was specified in transform.json, bypass search
  if (!customSpawn) {
    const supportStart = await findSupportedStartPosition(context, resolvedConfig, options);
    if (supportStart) {
      resolvedConfig = {
        ...resolvedConfig,
        startX: supportStart.x,
        startY: supportStart.y,
      };
      context.config = resolvedConfig;
      console.info(
        "[mjweb gaussian] support start",
        {
          tile: supportStart.tile,
          x: supportStart.x,
          y: supportStart.y,
          supportRatio: supportStart.supportRatio,
          heightRange: supportStart.heightRange,
          score: supportStart.score,
        },
      );
    }
  } else {
    console.info("[mjweb gaussian] using custom spawn from transform.json", customSpawn);
  }

  const [windowCenterX, windowCenterY] = initialHeightfieldWindowCenter(resolvedConfig);
  const heights = await sampleSupportHeightfieldWindow(
    context,
    windowCenterX,
    windowCenterY,
    options,
  );
  throwIfAborted(options.signal);
  clampHeightfield(heights, resolvedConfig.elevationScale);
  const startHeight = sampleStartHeight(
    heights,
    resolvedConfig.ncol,
    resolvedConfig.nrow,
    resolvedConfig.sizeX,
    resolvedConfig.sizeY,
    resolvedConfig.startX - windowCenterX,
    resolvedConfig.startY - windowCenterY,
    startHeightSampleRadiusForProjection(resolvedConfig, projection),
  );
  const baseProjection = splatProjectionMatrix(projection);
  const finalMatrix = multiplyMatrices4x4(baseProjection, loaded.transformMatrix);
  publishGaussianDebugState(loaded.source.name, splats.length, resolvedConfig, projection, startHeight, finalMatrix);
  const visualSource = {
    source: loaded.source,
    matrix: finalMatrix,
    boundsRadius: splatProjectionBoundsRadius(projection),
  };
  return {
    config: resolvedConfig,
    heights,
    startHeight,
    visualSource,
    windowCenterX,
    windowCenterY,
    dynamicHeightfield: resolvedConfig.dynamicWindow
      ? new DynamicGaussianHeightfield(context, windowCenterX, windowCenterY)
      : null,
    transformConfig: loaded.transformConfig,
    customSpawn,
  };
}

function createProceduralSlopeHeightfield(config: ProceduralSlopeHeightfieldConfig): PreparedHeightfield {
  const heights = createSmoothSlope(config);
  smoothHeightfield(heights, config.ncol, config.nrow, config.smoothingPasses);
  clampHeightfield(heights, config.elevationScale);
  const startHeight = sampleStartHeight(
    heights,
    config.ncol,
    config.nrow,
    config.sizeX,
    config.sizeY,
    config.startX,
    config.startY,
    config.startHeightSampleRadius ?? 0,
  );
  return {
    config,
    heights,
    startHeight,
    visualSource: null,
    windowCenterX: 0,
    windowCenterY: 0,
    dynamicHeightfield: null,
  };
}

function createSmoothSlope(config: ProceduralSlopeHeightfieldConfig): Float32Array {
  const heights = new Float32Array(config.nrow * config.ncol);
  for (let row = 0; row < config.nrow; row += 1) {
    for (let col = 0; col < config.ncol; col += 1) {
      const x = gridX(config, col);
      const y = gridY(config, row);
      const longitudinal = smoothStep(clamp01((x + config.sizeX) / (2 * config.sizeX)));
      const crossFall = 0.018 * Math.cos((Math.PI * y) / config.sizeY);
      heights[row * config.ncol + col] =
        config.slopeHeight * longitudinal + crossFall * longitudinal;
    }
  }
  return heights;
}

function createFlatHeightfield(config: BaseHeightfieldConfig): Float32Array {
  return new Float32Array(config.nrow * config.ncol);
}

async function createGaussianSupportHeightfieldContext(
  config: GaussianSplatHeightfieldConfig,
  sourcePacked: PackedSplatsForGpu,
  projection: SplatProjection,
  options: HeightfieldPreparationOptions,
): Promise<GaussianSupportHeightfieldContext> {
  throwIfAborted(options.signal);
  const device = await getSupportGpuDevice();
  throwIfAborted(options.signal);
  const packed = transformPackedSplatsToSupportHeightfield(sourcePacked, projection);
  const chunkSize = config.supportChunkSize ?? DEFAULT_SUPPORT_CHUNK_SIZE;
  const chunkResolution = Math.max(1, Math.floor(config.supportChunkResolution ?? DEFAULT_SUPPORT_CHUNK_RESOLUTION));
  const chunkPadding = config.supportChunkPadding ?? DEFAULT_SUPPORT_CHUNK_PADDING;
  const maxConcurrentJobs = Math.max(
    1,
    Math.floor(config.supportMaxConcurrentJobs ?? DEFAULT_SUPPORT_MAX_CONCURRENT_JOBS),
  );
  const binning = new SplatSpatialBinning(packed, chunkSize, chunkPadding);
  const manager = new ChunkVoxelManager(device, binning, chunkResolution, {
    maxConcurrentJobs,
    smoothIterations: config.supportSmoothIterations ?? DEFAULT_SUPPORT_SMOOTH_ITERATIONS,
    generationMode: "heightfield",
    supportFillMode: config.supportFillMode ?? "fallback",
    supportDensityThreshold: config.supportDensityThreshold ?? DEFAULT_SUPPORT_DENSITY_THRESHOLD,
    supportFillIterations: config.supportFillIterations ?? DEFAULT_SUPPORT_FILL_ITERATIONS,
    supportFillMaxHeightRange: config.supportFillMaxHeightRange ?? DEFAULT_SUPPORT_FILL_MAX_HEIGHT_RANGE,
  });
  manager.enableHeightfieldSmoothing = true;
  return {
    config,
    binning,
    manager,
    boundsMin: packed.boundsMin,
    boundsMax: packed.boundsMax,
    chunkSize,
    chunkResolution,
    maxConcurrentJobs,
  };
}

async function sampleSupportHeightfieldWindow(
  context: GaussianSupportHeightfieldContext,
  centerX: number,
  centerY: number,
  options: HeightfieldPreparationOptions,
  windowConfig: GaussianSplatHeightfieldConfig = context.config,
): Promise<Float32Array> {
  const { binning, manager, chunkSize, chunkResolution, maxConcurrentJobs } = context;
  const config = windowConfig;
  const fallback = createFlatHeightfield(config);
  const tileCoords = sortSupportTileCoordsForStart(
    supportColumnCoordsForWindow(config, binning, chunkSize, centerX, centerY),
    centerX,
    centerY,
    chunkSize,
  );
  const tiles = await loadSupportHeightTiles(
    manager,
    tileCoords,
    maxConcurrentJobs,
    options,
  );
  const heights = new Float32Array(fallback);
  options.onProgress?.({
    stage: "sampling",
    completed: 0,
    total: config.nrow,
    detail: `${config.ncol} x ${config.nrow} heightfield`,
  });
  for (let row = 0; row < config.nrow; row += 1) {
    throwIfAborted(options.signal);
    for (let col = 0; col < config.ncol; col += 1) {
      const index = row * config.ncol + col;
      heights[index] = sampleSupportHeight(
        tiles,
        centerX + gridX(config, col),
        centerY + gridY(config, row),
        chunkSize,
        chunkResolution,
        fallback[index],
        config.supportFillMode ?? "fallback",
      );
    }
    if ((row + 1) % HEIGHTFIELD_SAMPLE_PROGRESS_ROWS === 0 || row + 1 === config.nrow) {
      options.onProgress?.({
        stage: "sampling",
        completed: row + 1,
        total: config.nrow,
        detail: `${config.ncol} x ${config.nrow} heightfield`,
      });
      await yieldToMainThread();
    }
  }
  return heights;
}

function sortSupportTileCoordsForStart(
  tileCoords: [number, number][],
  centerX: number,
  centerY: number,
  chunkSize: number,
): [number, number][] {
  const startCx = Math.floor(centerX / chunkSize);
  const startCz = Math.floor(centerY / chunkSize);
  return [...tileCoords].sort((a, b) => {
    const da = squaredDistance2(a[0], a[1], startCx, startCz);
    const db = squaredDistance2(b[0], b[1], startCx, startCz);
    return da - db;
  });
}

async function loadSupportHeightTiles(
  manager: ChunkVoxelManager,
  tileCoords: [number, number][],
  concurrency: number,
  options: HeightfieldPreparationOptions,
): Promise<Map<string, HeightField>> {
  const tiles = new Map<string, HeightField>();
  let nextIndex = 0;
  let completed = 0;
  const workerCount = Math.min(concurrency, tileCoords.length);
  options.onProgress?.({
    stage: "support-tiles",
    completed,
    total: tileCoords.length,
    detail: `${workerCount} workers`,
  });
  const workers = Array.from({ length: workerCount }, async () => {
    while (nextIndex < tileCoords.length) {
      throwIfAborted(options.signal);
      const index = nextIndex;
      nextIndex += 1;
      const [cx, cz] = tileCoords[index];
      const tile = await manager.loadSupportHeightTileNow(cx, cz);
      throwIfAborted(options.signal);
      tiles.set(`${cx},${cz}`, tile);
      completed += 1;
      options.onProgress?.({
        stage: "support-tiles",
        completed,
        total: tileCoords.length,
        detail: `chunk ${cx}, ${cz}`,
      });
      await yieldToMainThread();
    }
  });
  await Promise.all(workers);
  return tiles;
}

function squaredDistance2(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

function supportColumnCoordsForWindow(
  config: GaussianSplatHeightfieldConfig,
  binning: SplatSpatialBinning,
  chunkSize: number,
  centerX: number,
  centerY: number,
): [number, number][] {
  const minX = centerX - config.sizeX;
  const maxX = centerX + config.sizeX;
  const minY = centerY - config.sizeY;
  const maxY = centerY + config.sizeY;
  return binning
    .getColumnCoordinatesByDensity(Number.MAX_SAFE_INTEGER)
    .filter(([cx, cz]) => {
      const tileMinX = cx * chunkSize;
      const tileMaxX = (cx + 1) * chunkSize;
      const tileMinY = cz * chunkSize;
      const tileMaxY = (cz + 1) * chunkSize;
      return tileMaxX >= minX && tileMinX <= maxX && tileMaxY >= minY && tileMinY <= maxY;
    });
}

function sampleSupportHeight(
  tiles: Map<string, HeightField>,
  x: number,
  y: number,
  chunkSize: number,
  chunkResolution: number,
  fallback: number,
  fillMode: NonNullable<GaussianSplatHeightfieldConfig["supportFillMode"]>,
): number {
  const voxelSize = chunkSize / chunkResolution;
  const gx = x / voxelSize - 0.5;
  const gy = y / voxelSize - 0.5;
  const gx0 = Math.floor(gx);
  const gy0 = Math.floor(gy);
  const tx = gx - gx0;
  const ty = gy - gy0;
  let sumHeight = 0;
  let sumWeight = 0;

  for (let dy = 0; dy <= 1; dy += 1) {
    for (let dx = 0; dx <= 1; dx += 1) {
      const sampleX = gx0 + dx;
      const sampleY = gy0 + dy;
      const tileCx = Math.floor(sampleX / chunkResolution);
      const tileCz = Math.floor(sampleY / chunkResolution);
      const tile = tiles.get(`${tileCx},${tileCz}`);
      if (!tile) {
        continue;
      }
      const vx = sampleX - tileCx * chunkResolution;
      const vy = sampleY - tileCz * chunkResolution;
      if (vx < 0 || vx >= tile.width || vy < 0 || vy >= tile.depth) {
        continue;
      }
      const height = resolveSupportHeight(tile, vx + vy * tile.width, fillMode);
      if (height === undefined) {
        continue;
      }
      const weightX = dx === 0 ? 1 - tx : tx;
      const weightY = dy === 0 ? 1 - ty : ty;
      const weight = weightX * weightY;
      sumHeight += height * weight;
      sumWeight += weight;
    }
  }

  return sumWeight > 0.0001 ? sumHeight / sumWeight : fallback;
}

function resolveSupportHeight(
  field: HeightField,
  index: number,
  fillMode: NonNullable<GaussianSplatHeightfieldConfig["supportFillMode"]>,
): number | undefined {
  const fallbackHeight = Number.isFinite(field.fallbackHeight) ? field.fallbackHeight : undefined;
  const canUseFallback = fallbackHeight !== undefined && (fillMode === "fallback" || fillMode === "min");
  if (field.state[index] === 0) {
    return canUseFallback ? fallbackHeight : undefined;
  }
  const height = field.height[index];
  return Number.isFinite(height) ? height : canUseFallback ? fallbackHeight : undefined;
}

function computeSplatProjection(
  config: GaussianSplatHeightfieldConfig,
  splats: GaussianSplat[],
  sourceBounds?: SplatBounds,
): SplatProjection {
  const bounds = sourceBounds ?? computeBounds(splats);
  const rangeX = Math.max(1e-4, bounds.max[0] - bounds.min[0]);
  const rangeY = Math.max(1e-4, bounds.max[1] - bounds.min[1]);
  const rangeZ = Math.max(1e-4, bounds.max[2] - bounds.min[2]);
  const sourceScaleMultiplier = Math.max(1e-3, config.sourceScaleMultiplier ?? 1);
  const fitMargin = 0.92;
  const fitScale = fitMargin * Math.min((2 * config.sizeX) / rangeX, (2 * config.sizeY) / rangeZ);
  const metricScale = Math.max(1e-6, config.sourceMetersPerUnit ?? 1);
  const metricMode = config.sourceScaleMode === "metric";
  const horizontalScale = sourceScaleMultiplier * (metricMode ? metricScale : fitScale);
  const fitHeight = Math.max(1e-6, config.sourceFitHeight ?? config.elevationScale);
  const heightScale = sourceScaleMultiplier * (metricMode ? metricScale : fitHeight / rangeY);
  const lateralSign = 1;
  const verticalSign = 1;
  const center = computeProjectionCenter(config, bounds, splats);
  const centerX = center[0];
  const centerZ = center[1];
  const footprintMinX = (bounds.min[0] - centerX) * horizontalScale;
  const footprintMaxX = (bounds.max[0] - centerX) * horizontalScale;
  const projectedMinY = (bounds.min[2] - centerZ) * horizontalScale * lateralSign;
  const projectedMaxY = (bounds.max[2] - centerZ) * horizontalScale * lateralSign;
  const footprintMinY = Math.min(projectedMinY, projectedMaxY);
  const footprintMaxY = Math.max(projectedMinY, projectedMaxY);
  return {
    boundsMin: [...bounds.min] as Vec3,
    boundsMax: [...bounds.max] as Vec3,
    rangeY,
    horizontalScale,
    heightScale,
    lateralSign,
    verticalSign,
    centerX,
    centerZ,
    minY: bounds.min[1],
    maxY: bounds.max[1],
    maxHeight: rangeY * heightScale,
    footprintMinX,
    footprintMaxX,
    footprintMinY,
    footprintMaxY,
  };
}

function computeProjectionCenter(
  config: GaussianSplatHeightfieldConfig,
  bounds: ReturnType<typeof computeBounds>,
  splats: GaussianSplat[],
): [number, number] {
  if (config.sourceCenter === "origin") {
    return [0, 0];
  }
  if (config.sourceCenter === "density") {
    return denseHorizontalCenter(bounds, splats);
  }
  return [
    0.5 * (bounds.min[0] + bounds.max[0]),
    0.5 * (bounds.min[2] + bounds.max[2]),
  ];
}

function denseHorizontalCenter(
  bounds: ReturnType<typeof computeBounds>,
  splats: GaussianSplat[],
): [number, number] {
  const rangeX = Math.max(1e-4, bounds.max[0] - bounds.min[0]);
  const rangeZ = Math.max(1e-4, bounds.max[2] - bounds.min[2]);
  const gridSize = 128;
  const neighborhoodRadius = 2;
  const counts = new Uint32Array(gridSize * gridSize);

  for (const splat of splats) {
    if (splat.opacity < 0.03) {
      continue;
    }
    const gx = Math.min(
      gridSize - 1,
      Math.max(0, Math.floor(((splat.center[0] - bounds.min[0]) / rangeX) * gridSize)),
    );
    const gz = Math.min(
      gridSize - 1,
      Math.max(0, Math.floor(((splat.center[2] - bounds.min[2]) / rangeZ) * gridSize)),
    );
    counts[gz * gridSize + gx] += 1;
  }

  let bestIndex = 0;
  for (let i = 1; i < counts.length; i += 1) {
    if (counts[i] > counts[bestIndex]) {
      bestIndex = i;
    }
  }
  if (counts[bestIndex] === 0) {
    return [
      0.5 * (bounds.min[0] + bounds.max[0]),
      0.5 * (bounds.min[2] + bounds.max[2]),
    ];
  }

  const bestX = bestIndex % gridSize;
  const bestZ = Math.floor(bestIndex / gridSize);
  const minGridX = Math.max(0, bestX - neighborhoodRadius);
  const maxGridX = Math.min(gridSize - 1, bestX + neighborhoodRadius);
  const minGridZ = Math.max(0, bestZ - neighborhoodRadius);
  const maxGridZ = Math.min(gridSize - 1, bestZ + neighborhoodRadius);
  const minX = bounds.min[0] + (minGridX / gridSize) * rangeX;
  const maxX = bounds.min[0] + ((maxGridX + 1) / gridSize) * rangeX;
  const minZ = bounds.min[2] + (minGridZ / gridSize) * rangeZ;
  const maxZ = bounds.min[2] + ((maxGridZ + 1) / gridSize) * rangeZ;
  let count = 0;
  let sumX = 0;
  let sumZ = 0;

  for (const splat of splats) {
    if (splat.opacity < 0.03) {
      continue;
    }
    const x = splat.center[0];
    const z = splat.center[2];
    if (x < minX || x > maxX || z < minZ || z > maxZ) {
      continue;
    }
    count += 1;
    sumX += x;
    sumZ += z;
  }

  if (count === 0) {
    return [
      bounds.min[0] + ((bestX + 0.5) / gridSize) * rangeX,
      bounds.min[2] + ((bestZ + 0.5) / gridSize) * rangeZ,
    ];
  }
  return [sumX / count, sumZ / count];
}

function resolutionForMetricProjection(
  config: GaussianSplatHeightfieldConfig,
  sizeX: number,
  sizeY: number,
): [number, number] {
  const targetCellSize = config.sourceTargetCellSize;
  if (!targetCellSize || targetCellSize <= 0) {
    return [config.nrow, config.ncol];
  }
  const maxRows = Math.max(config.nrow, config.sourceMaxRows ?? config.nrow);
  const maxCols = Math.max(config.ncol, config.sourceMaxCols ?? config.ncol);
  const nrow = clampInteger(
    Math.ceil((2 * sizeY) / targetCellSize) + 1,
    config.nrow,
    maxRows,
  );
  const ncol = clampInteger(
    Math.ceil((2 * sizeX) / targetCellSize) + 1,
    config.ncol,
    maxCols,
  );
  return [nrow, ncol];
}

function configForProjection(
  config: GaussianSplatHeightfieldConfig,
  projection: SplatProjection,
): GaussianSplatHeightfieldConfig {
  const padding = Math.max(0, config.sourceBoundsPadding ?? 1);
  const metricMode = config.sourceScaleMode === "metric";
  const dynamicWindow = config.dynamicWindow === true;
  const sizeX = metricMode && !dynamicWindow
    ? Math.max(
        config.sizeX,
        Math.abs(projection.footprintMinX) + padding,
        Math.abs(projection.footprintMaxX) + padding,
      )
    : config.sizeX;
  const sizeY = metricMode && !dynamicWindow
    ? Math.max(
        config.sizeY,
        Math.abs(projection.footprintMinY) + padding,
        Math.abs(projection.footprintMaxY) + padding,
      )
    : config.sizeY;
  const cappedSizeX = metricMode && config.sourceMaxSizeX
    ? Math.min(sizeX, Math.max(config.sizeX, config.sourceMaxSizeX))
    : sizeX;
  const cappedSizeY = metricMode && config.sourceMaxSizeY
    ? Math.min(sizeY, Math.max(config.sizeY, config.sourceMaxSizeY))
    : sizeY;
  const elevationScale = metricMode
    ? Math.max(config.elevationScale, projection.maxHeight + 0.1)
    : config.elevationScale;
  const resolvedSizeX = metricMode ? roundUp(cappedSizeX, 0.5) : sizeX;
  const resolvedSizeY = metricMode ? roundUp(cappedSizeY, 0.5) : sizeY;
  const resolvedElevationScale = metricMode ? roundUp(elevationScale, 0.1) : elevationScale;
  const [resolvedNrow, resolvedNcol] = metricMode
    ? resolutionForMetricProjection(config, resolvedSizeX, resolvedSizeY)
    : [config.nrow, config.ncol];
  const startX = config.startAtGaussianOrigin
    ? 0
    : config.startX;
  const startY = config.startAtGaussianOrigin
    ? 0
    : config.startYOnSourceFootprint
      ? startYOnFootprint(config.startY, projection, resolvedSizeY)
      : config.startY;

  if (
    resolvedSizeX === config.sizeX &&
    resolvedSizeY === config.sizeY &&
    resolvedElevationScale === config.elevationScale &&
    resolvedNrow === config.nrow &&
    resolvedNcol === config.ncol &&
    startX === config.startX &&
    startY === config.startY
  ) {
    return config;
  }

  return {
    ...config,
    nrow: resolvedNrow,
    ncol: resolvedNcol,
    sizeX: resolvedSizeX,
    sizeY: resolvedSizeY,
    elevationScale: resolvedElevationScale,
    startX,
    startY,
  };
}

function initialHeightfieldWindowCenter(config: GaussianSplatHeightfieldConfig): [number, number] {
  return config.dynamicWindow ? [config.startX, config.startY] : [0, 0];
}

async function findSupportedStartPosition(
  context: GaussianSupportHeightfieldContext,
  config: GaussianSplatHeightfieldConfig,
  options: HeightfieldPreparationOptions,
): Promise<SupportStartCandidate | null> {
  if (config.startPlacement !== "support") {
    return null;
  }
  const limit = Math.max(
    1,
    Math.floor(config.startSearchChunkLimit ?? DEFAULT_START_SEARCH_CHUNK_LIMIT),
  );
  const tileCoords = context.binning.getColumnCoordinatesByDensity(limit);
  let best: SupportStartCandidate | null = null;

  options.onProgress?.({
    stage: "support-tiles",
    completed: 0,
    total: tileCoords.length,
    detail: "spawn search",
  });
  for (let i = 0; i < tileCoords.length; i += 1) {
    throwIfAborted(options.signal);
    const tileCoordsForCandidate = tileCoords[i];
    const field = await context.manager.loadSupportHeightTileNow(
      tileCoordsForCandidate[0],
      tileCoordsForCandidate[1],
    );
    const candidate = findBestStartInSupportTile(
      field,
      tileCoordsForCandidate,
      context.chunkSize,
      config,
    );
    if (candidate && (!best || candidate.score > best.score)) {
      best = candidate;
    }
    options.onProgress?.({
      stage: "support-tiles",
      completed: i + 1,
      total: tileCoords.length,
      detail: `spawn ${tileCoordsForCandidate[0]}, ${tileCoordsForCandidate[1]}`,
    });
    await yieldToMainThread();
  }

  if (best) {
    return best;
  }

  const fallback = findDensestSplatStartPosition(context);
  return fallback
    ? {
        ...fallback,
        score: Number.NEGATIVE_INFINITY,
        supportRatio: 0,
        heightRange: Number.POSITIVE_INFINITY,
      }
    : null;
}

function findBestStartInSupportTile(
  field: HeightField,
  tileCoords: [number, number],
  chunkSize: number,
  config: GaussianSplatHeightfieldConfig,
): SupportStartCandidate | null {
  const dimX = field.width;
  const dimY = field.depth;
  const voxelSize = chunkSize / Math.max(1, dimX);
  const supportRadius = Math.max(0, config.startSupportRadius ?? DEFAULT_START_SUPPORT_RADIUS);
  const radiusCells = Math.max(1, Math.ceil(supportRadius / Math.max(1e-6, voxelSize)));
  const radiusSq = (supportRadius / Math.max(1e-6, voxelSize)) ** 2;
  const minSupportRatio = clampNumber(
    config.startMinSupportRatio ?? DEFAULT_START_MIN_SUPPORT_RATIO,
    0,
    1,
  );
  const maxHeightRange = Math.max(
    0,
    config.startMaxHeightRange ?? DEFAULT_START_MAX_HEIGHT_RANGE,
  );
  let best: SupportStartCandidate | null = null;

  for (let row = radiusCells; row < dimY - radiusCells; row += 1) {
    for (let col = radiusCells; col < dimX - radiusCells; col += 1) {
      const centerIndex = col + row * dimX;
      const centerHeight = field.height[centerIndex];
      if (field.state[centerIndex] === 0 || !Number.isFinite(centerHeight)) {
        continue;
      }

      let totalSamples = 0;
      let supportSamples = 0;
      let originalSamples = 0;
      let minSupportHeight = Number.POSITIVE_INFINITY;
      let maxSupportHeight = Number.NEGATIVE_INFINITY;

      for (let dy = -radiusCells; dy <= radiusCells; dy += 1) {
        for (let dx = -radiusCells; dx <= radiusCells; dx += 1) {
          if (dx * dx + dy * dy > radiusSq) {
            continue;
          }
          totalSamples += 1;

          const sampleIndex = col + dx + (row + dy) * dimX;
          const sampleHeight = field.height[sampleIndex];
          if (
            field.state[sampleIndex] === 0 ||
            !Number.isFinite(sampleHeight) ||
            Math.abs(sampleHeight - centerHeight) > maxHeightRange
          ) {
            continue;
          }

          supportSamples += 1;
          if (field.state[sampleIndex] === 1) {
            originalSamples += 1;
          }
          minSupportHeight = Math.min(minSupportHeight, sampleHeight);
          maxSupportHeight = Math.max(maxSupportHeight, sampleHeight);
        }
      }

      const supportRatio = supportSamples / Math.max(1, totalSamples);
      if (supportRatio < minSupportRatio) {
        continue;
      }

      const heightRange = maxSupportHeight - minSupportHeight;
      const originalRatio = originalSamples / Math.max(1, supportSamples);
      const edgeDistance = Math.min(col, row, dimX - 1 - col, dimY - 1 - row);
      const score =
        supportRatio * 1000 +
        originalRatio * 180 -
        heightRange * 300 +
        edgeDistance * 0.5;

      if (best && score <= best.score) {
        continue;
      }

      best = {
        x: tileCoords[0] * chunkSize + (col + 0.5) * voxelSize,
        y: tileCoords[1] * chunkSize + (row + 0.5) * voxelSize,
        score,
        supportRatio,
        heightRange,
        tile: tileCoords,
      };
    }
  }

  return best;
}

function findDensestSplatStartPosition(
  context: GaussianSupportHeightfieldContext,
): Pick<SupportStartCandidate, "x" | "y" | "tile"> | null {
  const [chunkX, chunkY, chunkZ] = context.binning.getDensestChunkCoordinates();
  const subset = context.binning.getSplatSubsetForChunk(chunkX, chunkY, chunkZ);
  if (!subset || subset.count <= 0) {
    return null;
  }
  let bestHeight = Number.NEGATIVE_INFINITY;
  let x = (chunkX + 0.5) * context.chunkSize;
  let y = (chunkZ + 0.5) * context.chunkSize;

  for (let i = 0; i < subset.count; i += 1) {
    const base = i * 4;
    const height = subset.centers[base + 1];
    if (height <= bestHeight) {
      continue;
    }
    bestHeight = height;
    x = subset.centers[base];
    y = subset.centers[base + 2];
  }

  return {
    x,
    y,
    tile: [chunkX, chunkZ],
  };
}

function routePlanningWindowConfig(
  config: GaussianSplatHeightfieldConfig,
  start: readonly [number, number],
  goal: readonly [number, number],
): GaussianSplatHeightfieldConfig {
  const padding = Math.max(2, Math.min(config.sizeX, config.sizeY));
  const halfX = Math.max(config.sizeX, 0.5 * Math.abs(goal[0] - start[0]) + padding);
  const halfY = Math.max(config.sizeY, 0.5 * Math.abs(goal[1] - start[1]) + padding);
  const sizeX = roundUp(halfX, 0.5);
  const sizeY = roundUp(halfY, 0.5);
  const [nrow, ncol] = resolutionForMetricProjection(config, sizeX, sizeY);
  return {
    ...config,
    dynamicWindow: false,
    sizeX,
    sizeY,
    nrow,
    ncol,
  };
}

function globalPlanningWindowConfig(
  config: GaussianSplatHeightfieldConfig,
  boundsMin: Vec3,
  boundsMax: Vec3,
): {
  config: GaussianSplatHeightfieldConfig;
  centerX: number;
  centerY: number;
} {
  const padding = Math.max(0, config.sourceBoundsPadding ?? 1);
  const centerX = 0.5 * (boundsMin[0] + boundsMax[0]);
  const centerY = 0.5 * (boundsMin[2] + boundsMax[2]);
  const sizeX = roundUp(Math.max(config.sizeX, 0.5 * (boundsMax[0] - boundsMin[0]) + padding), 0.5);
  const sizeY = roundUp(Math.max(config.sizeY, 0.5 * (boundsMax[2] - boundsMin[2]) + padding), 0.5);
  const [nrow, ncol] = resolutionForMetricProjection(config, sizeX, sizeY);
  return {
    config: {
      ...config,
      dynamicWindow: false,
      sizeX,
      sizeY,
      nrow,
      ncol,
    },
    centerX,
    centerY,
  };
}

function splatProjectionMatrix(projection: SplatProjection, visualLift = SPLAT_VISUAL_LIFT): number[] {
  return [
    projection.horizontalScale, 0, 0, -projection.horizontalScale * projection.centerX,
    0,
    0,
    projection.horizontalScale * projection.lateralSign,
    -projection.horizontalScale * projection.lateralSign * projection.centerZ,
    0,
    projection.heightScale * projection.verticalSign,
    0,
    verticalOffsetForProjection(projection) + visualLift,
    0, 0, 0, 1,
  ];
}

function transformSourcePoint(matrix: number[], point: Vec3): Vec3 {
  return [
    matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
    matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
    matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
  ];
}

function splatProjectionBoundsRadius(projection: SplatProjection): number {
  const maxX = Math.max(Math.abs(projection.footprintMinX), Math.abs(projection.footprintMaxX));
  const maxY = Math.max(Math.abs(projection.footprintMinY), Math.abs(projection.footprintMaxY));
  return Math.hypot(maxX, maxY, projection.maxHeight);
}

function publishGaussianDebugState(
  sourceName: string,
  splatCount: number,
  config: GaussianSplatHeightfieldConfig,
  projection: SplatProjection,
  startHeight: number,
  matrixOverride?: number[],
): void {
  if (typeof window === "undefined") {
    return;
  }
  const debugState: GaussianDebugState = {
    sourceName,
    splatCount,
    boundsMin: projection.boundsMin,
    boundsMax: projection.boundsMax,
    center: [projection.centerX, projection.centerZ],
    horizontalScale: projection.horizontalScale,
    heightScale: projection.heightScale,
    lateralSign: projection.lateralSign,
    verticalSign: projection.verticalSign,
    footprintMin: [projection.footprintMinX, projection.footprintMinY],
    footprintMax: [projection.footprintMaxX, projection.footprintMaxY],
    maxHeight: projection.maxHeight,
    hfield: {
      nrow: config.nrow,
      ncol: config.ncol,
      sizeX: config.sizeX,
      sizeY: config.sizeY,
      elevationScale: config.elevationScale,
      startX: config.startX,
      startY: config.startY,
      startHeight,
    },
    matrix: matrixOverride ?? splatProjectionMatrix(projection),
    boundsRadius: splatProjectionBoundsRadius(projection),
  };
  window.__mjwebGaussianDebug = debugState;
  document.documentElement.dataset.mjwebGaussianDebug = JSON.stringify(debugState);
  console.info("[mjweb gaussian]", debugState);
}

function verticalOffsetForProjection(projection: SplatProjection): number {
  return projection.verticalSign > 0
    ? -projection.minY * projection.heightScale
    : projection.maxY * projection.heightScale;
}

function startYOnFootprint(
  currentY: number,
  projection: SplatProjection,
  sizeY: number,
): number {
  const margin = Math.min(0.25, Math.max(0, sizeY * 0.1));
  const terrainMinY = -sizeY + margin;
  const terrainMaxY = sizeY - margin;
  const minY = Math.max(terrainMinY, projection.footprintMinY);
  const maxY = Math.min(terrainMaxY, projection.footprintMaxY);
  if (minY <= maxY) {
    if (currentY >= minY && currentY <= maxY) {
      return currentY;
    }
    return 0.5 * (minY + maxY);
  }
  const footprintCenterY = 0.5 * (projection.footprintMinY + projection.footprintMaxY);
  return clampNumber(footprintCenterY, terrainMinY, terrainMaxY);
}

function startHeightSampleRadiusForProjection(
  config: GaussianSplatHeightfieldConfig,
  projection: SplatProjection,
): number {
  const configured = config.startHeightSampleRadius ?? 0;
  if (!config.startAtGaussianOrigin) {
    return configured;
  }
  return Math.max(configured, Math.min(8, 0.4 * projection.horizontalScale));
}

function sourceNeedsColmapAxisFlip(sourceName?: string): boolean {
  if (!sourceName) {
    return false;
  }
  const lower = sourceName.toLowerCase();
  return (
    lower.endsWith(".sog") ||
    lower.endsWith(".spz") ||
    lower.endsWith(".ply") ||
    lower.endsWith(".splat")
  );
}


let supportGpuDevicePromise: Promise<any> | null = null;

async function getSupportGpuDevice(): Promise<any> {
  if (!supportGpuDevicePromise) {
    supportGpuDevicePromise = (async () => {
      const gpu = typeof navigator === "undefined"
        ? undefined
        : (navigator as Navigator & { gpu?: { requestAdapter(): Promise<any> } }).gpu;
      if (!gpu) {
        throw new Error("WebGPU is required for Gaussian support heightfields");
      }
      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        throw new Error("No WebGPU adapter is available for Gaussian support heightfields");
      }
      return adapter.requestDevice();
    })();
  }
  return supportGpuDevicePromise;
}

function packGaussianSplatsForGpu(splats: GaussianSplat[]): PackedSplatsForGpu {
  const centers = new Float32Array(splats.length * 4);
  const scales = new Float32Array(splats.length * 4);
  const quaternions = new Float32Array(splats.length * 4);
  const colors = new Float32Array(splats.length * 4);
  const boundsMin: Vec3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const boundsMax: Vec3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  for (let i = 0; i < splats.length; i += 1) {
    const splat = splats[i];
    const base = i * 4;
    centers[base] = splat.center[0];
    centers[base + 1] = splat.center[1];
    centers[base + 2] = splat.center[2];
    centers[base + 3] = 1;
    scales[base] = Math.max(1e-6, splat.scales[0]);
    scales[base + 1] = Math.max(1e-6, splat.scales[1]);
    scales[base + 2] = Math.max(1e-6, splat.scales[2]);
    scales[base + 3] = 0;
    quaternions[base] = splat.rotation[1];
    quaternions[base + 1] = splat.rotation[2];
    quaternions[base + 2] = splat.rotation[3];
    quaternions[base + 3] = splat.rotation[0];
    colors[base] = splat.color[0];
    colors[base + 1] = splat.color[1];
    colors[base + 2] = splat.color[2];
    colors[base + 3] = splat.opacity;
    for (let axis = 0; axis < 3; axis += 1) {
      boundsMin[axis] = Math.min(boundsMin[axis], splat.center[axis]);
      boundsMax[axis] = Math.max(boundsMax[axis], splat.center[axis]);
    }
  }

  return {
    centers,
    scales,
    quaternions,
    colors,
    boundsMin,
    boundsMax,
    count: splats.length,
  };
}

function transformPackedSplatsToSupportHeightfield(
  packed: PackedSplatsForGpu,
  projection: SplatProjection,
): PackedSplatsForGpu {
  const sourceToHeightfield = splatProjectionMatrix(projection, 0);
  const centers = new Float32Array(packed.centers.length);
  const scales = new Float32Array(packed.scales.length);
  const quaternions = new Float32Array(packed.quaternions);
  const colors = new Float32Array(packed.colors);
  const boundsMin: Vec3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const boundsMax: Vec3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const horizontalScale = Math.abs(projection.horizontalScale);
  const heightScale = Math.abs(projection.heightScale);

  for (let i = 0; i < packed.count; i += 1) {
    const base = i * 4;
    const [x, y, height] = transformSourcePoint(sourceToHeightfield, [
      packed.centers[base],
      packed.centers[base + 1],
      packed.centers[base + 2],
    ]);
    centers[base] = x;
    centers[base + 1] = height;
    centers[base + 2] = y;
    centers[base + 3] = 1;
    scales[base] = Math.max(1e-6, packed.scales[base] * horizontalScale);
    scales[base + 1] = Math.max(1e-6, packed.scales[base + 1] * heightScale);
    scales[base + 2] = Math.max(1e-6, packed.scales[base + 2] * horizontalScale);
    scales[base + 3] = 0;
    boundsMin[0] = Math.min(boundsMin[0], x);
    boundsMin[1] = Math.min(boundsMin[1], height);
    boundsMin[2] = Math.min(boundsMin[2], y);
    boundsMax[0] = Math.max(boundsMax[0], x);
    boundsMax[1] = Math.max(boundsMax[1], height);
    boundsMax[2] = Math.max(boundsMax[2], y);
  }

  return {
    centers,
    scales,
    quaternions,
    colors,
    boundsMin,
    boundsMax,
    count: packed.count,
  };
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) {
    return;
  }
  const error = new Error("Heightfield generation aborted");
  error.name = "AbortError";
  throw error;
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => globalThis.setTimeout(resolve, 0));
}

interface LoadedGaussianSource {
  source: GaussianHeightfieldSource;
  splats: GaussianSplat[];
  bounds?: SplatBounds;
  transformMatrix: number[];
  transformConfig?: TransformConfig | null;
}

interface TransformConfig {
  matrix?: number[] | Record<string, number[]>;
  scale?: number;
  spawn?: [number, number] | [number, number, number] | { x: number; y: number; yaw?: number };
}

const IDENTITY_MATRIX = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

const COLMAP_FLIP_MATRIX = [
  1,  0,  0, 0,
  0, -1,  0, 0,
  0,  0, -1, 0,
  0,  0,  0, 1,
];

function transformPoint(matrix: number[], point: Vec3): Vec3 {
  return [
    matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
    matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
    matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
  ];
}

function multiplyMatrices4x4(a: number[], b: number[]): number[] {
  const out = new Array<number>(16);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      out[r * 4 + c] =
        a[r * 4 + 0] * b[0 * 4 + c] +
        a[r * 4 + 1] * b[1 * 4 + c] +
        a[r * 4 + 2] * b[2 * 4 + c] +
        a[r * 4 + 3] * b[3 * 4 + c];
    }
  }
  return out;
}

async function tryFetchTransformConfig(sourceUrl: string): Promise<TransformConfig | null> {
  try {
    const lastSlash = sourceUrl.lastIndexOf("/");
    if (lastSlash < 0) {
      return null;
    }
    const dir = sourceUrl.substring(0, lastSlash);
    const transformUrl = `${dir}/transform.json`;
    const response = await fetch(transformUrl);
    if (!response.ok) {
      return null;
    }
    const json = await response.json();
    return json as TransformConfig;
  } catch {
    return null;
  }
}

function getMatrixFromConfig(config: TransformConfig, fileName: string): number[] | undefined {
  if (config.matrix === undefined) {
    return undefined;
  }
  if (Array.isArray(config.matrix) && config.matrix.length === 16) {
    return config.matrix;
  }
  if (typeof config.matrix === "object" && config.matrix !== null) {
    const val = (config.matrix as Record<string, number[]>)[fileName];
    if (Array.isArray(val) && val.length === 16) {
      return val;
    }
  }
  return undefined;
}

function defaultTransformMatrix(sourceName?: string): number[] {
  if (sourceNeedsColmapAxisFlip(sourceName)) {
    return COLMAP_FLIP_MATRIX;
  }
  return IDENTITY_MATRIX;
}

function bakeGaussianSource(
  loadedSet: GaussianSplatSet,
  fileName: string,
  transformMatrix: number[],
): LoadedGaussianSource {
  const splats = loadedSet.splats;
  for (const splat of splats) {
    splat.center = transformPoint(transformMatrix, splat.center);
  }
  const bounds = computeBounds(splats);
  return {
    source: { name: fileName, bytes: new Uint8Array() },
    splats,
    bounds,
    transformMatrix,
  };
}

async function loadFirstGaussianSource(
  config: GaussianSplatHeightfieldConfig,
  envId?: string,
): Promise<LoadedGaussianSource | null> {
  const sourceUrls = [
    ...gaussianSourceUrlsForBundle(envId, config.sourceBundleId),
    ...(config.sourceUrls ?? []),
  ];
  for (const sourceUrl of sourceUrls) {
    const bytes = await tryFetchBytes(sourceUrl);
    if (!bytes) {
      continue;
    }
    const name = sourceUrl.split("/").pop() || sourceUrl;
    const loadedSet = await loadGaussianSource(sourceUrl, bytes);
    const transformConfig = await tryFetchTransformConfig(sourceUrl);
    const matrix = (transformConfig ? getMatrixFromConfig(transformConfig, name) : undefined)
      ?? defaultTransformMatrix(name);
    
    const baked = bakeGaussianSource(loadedSet, name, matrix);
    baked.source.bytes = bytes;
    baked.transformConfig = transformConfig;
    return baked;
  }
  return null;
}

async function loadGaussianSource(fileName: string, bytes: Uint8Array): Promise<GaussianSplatSet> {
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".splat")) {
    return { splats: parseSplat(bytes) };
  }
  if (lower.endsWith(".ply")) {
    return { splats: parsePlyPositions(bytes) };
  }
  if (lower.endsWith(".spz") || lower.endsWith(".sog")) {
    return parseSparkGaussian(fileName, bytes);
  }
  throw new Error(`Unsupported Gaussian source: ${fileName}`);
}

async function parseSparkGaussian(fileName: string, bytes: Uint8Array): Promise<GaussianSplatSet> {
  const { SplatMesh } = await import("@sparkjsdev/spark");
  const mesh = new SplatMesh({
    fileBytes: bytes,
    fileName,
    extSplats: true,
  });
  await mesh.initialized;
  try {
    const packed = packSplats(mesh, mesh.numSplats, 0.01);
    const splats: GaussianSplat[] = [];
    mesh.forEachSplat((_, center, scales, quaternion, opacity, color) => {
      splats.push({
        center: [center.x, center.y, center.z],
        scales: [scales.x, scales.y, scales.z],
        rotation: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
        opacity,
        color: [
          Math.round(color.r * 255),
          Math.round(color.g * 255),
          Math.round(color.b * 255),
        ],
      });
    });
    return {
      splats,
      bounds: {
        min: [...packed.boundsMin] as Vec3,
        max: [...packed.boundsMax] as Vec3,
      },
      packed,
    };
  } finally {
    mesh.dispose();
  }
}

async function tryFetchBytes(url: string): Promise<Uint8Array | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("text/html")) {
      return null;
    }
    return new Uint8Array(await response.arrayBuffer());
  } catch {
    return null;
  }
}

function parseSplat(bytes: Uint8Array): GaussianSplat[] {
  const stride = 32;
  if (bytes.byteLength % stride !== 0) {
    throw new Error(`Invalid .splat file size ${bytes.byteLength}; expected a multiple of ${stride}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const splats: GaussianSplat[] = [];
  for (let i = 0; i < bytes.byteLength / stride; i += 1) {
    const base = i * stride;
    splats.push({
      center: [
        view.getFloat32(base, true),
        view.getFloat32(base + 4, true),
        view.getFloat32(base + 8, true),
      ],
      scales: [
        view.getFloat32(base + 12, true),
        view.getFloat32(base + 16, true),
        view.getFloat32(base + 20, true),
      ],
      color: [view.getUint8(base + 24), view.getUint8(base + 25), view.getUint8(base + 26)],
      opacity: view.getUint8(base + 27) / 255,
      rotation: normalizeQuat([
        (view.getUint8(base + 28) - 128) / 128,
        (view.getUint8(base + 29) - 128) / 128,
        (view.getUint8(base + 30) - 128) / 128,
        (view.getUint8(base + 31) - 128) / 128,
      ]),
    });
  }
  return splats;
}

function parsePlyPositions(bytes: Uint8Array): GaussianSplat[] {
  const header = parsePlyHeader(bytes);
  const xIndex = header.vertexProperties.findIndex((prop) => prop.name === "x");
  const yIndex = header.vertexProperties.findIndex((prop) => prop.name === "y");
  const zIndex = header.vertexProperties.findIndex((prop) => prop.name === "z");
  if (xIndex < 0 || yIndex < 0 || zIndex < 0) {
    throw new Error("Invalid PLY: vertex x/y/z properties are required");
  }
  return header.format === "ascii"
    ? parseAsciiPlyVertices(bytes, header, xIndex, yIndex, zIndex)
    : parseBinaryPlyVertices(bytes, header, xIndex, yIndex, zIndex);
}

function parsePlyHeader(bytes: Uint8Array): PlyHeader {
  const headerEnd = findPlyHeaderEnd(bytes);
  const text = new TextDecoder().decode(bytes.subarray(0, headerEnd));
  const lines = text.split(/\r?\n/);
  let format: PlyHeader["format"] | null = null;
  let vertexCount = 0;
  let inVertex = false;
  const vertexProperties: PlyProperty[] = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "format") {
      if (parts[1] !== "ascii" && parts[1] !== "binary_little_endian") {
        throw new Error(`Unsupported PLY format: ${parts[1]}`);
      }
      format = parts[1];
    } else if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      if (inVertex) {
        vertexCount = Number(parts[2]);
      }
    } else if (parts[0] === "property" && inVertex) {
      if (parts[1] === "list") {
        throw new Error("PLY list properties on vertices are not supported");
      }
      vertexProperties.push({ type: normalizePlyType(parts[1]), name: parts[2] });
    }
  }

  if (!format) {
    throw new Error("Invalid PLY: missing format");
  }
  return { format, vertexCount, vertexProperties, headerEnd };
}

function parseAsciiPlyVertices(
  bytes: Uint8Array,
  header: PlyHeader,
  xIndex: number,
  yIndex: number,
  zIndex: number,
): GaussianSplat[] {
  const text = new TextDecoder().decode(bytes.subarray(header.headerEnd));
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const splats: GaussianSplat[] = [];
  for (let i = 0; i < Math.min(header.vertexCount, lines.length); i += 1) {
    const parts = lines[i].trim().split(/\s+/).map(Number);
    splats.push(makePointSplat(parts[xIndex], parts[yIndex], parts[zIndex]));
  }
  return splats;
}

function parseBinaryPlyVertices(
  bytes: Uint8Array,
  header: PlyHeader,
  xIndex: number,
  yIndex: number,
  zIndex: number,
): GaussianSplat[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const stride = header.vertexProperties.reduce((sum, prop) => sum + plyScalarSize(prop.type), 0);
  const offsets = header.vertexProperties.map((_, index) =>
    header.vertexProperties
      .slice(0, index)
      .reduce((sum, prop) => sum + plyScalarSize(prop.type), 0),
  );
  const splats: GaussianSplat[] = [];
  for (let i = 0; i < header.vertexCount; i += 1) {
    const base = header.headerEnd + i * stride;
    if (base + stride > bytes.byteLength) {
      break;
    }
    splats.push(makePointSplat(
      readPlyScalar(view, base + offsets[xIndex], header.vertexProperties[xIndex].type),
      readPlyScalar(view, base + offsets[yIndex], header.vertexProperties[yIndex].type),
      readPlyScalar(view, base + offsets[zIndex], header.vertexProperties[zIndex].type),
    ));
  }
  return splats;
}

function makePointSplat(x: number, y: number, z: number): GaussianSplat {
  return {
    center: [x, y, z],
    scales: [0.01, 0.01, 0.01],
    rotation: [1, 0, 0, 0],
    opacity: 1,
    color: [220, 220, 220],
  };
}

function findPlyHeaderEnd(bytes: Uint8Array): number {
  const marker = new TextEncoder().encode("end_header\n");
  const crlfMarker = new TextEncoder().encode("end_header\r\n");
  const lf = indexOfBytes(bytes, marker);
  if (lf >= 0) {
    return lf + marker.length;
  }
  const crlf = indexOfBytes(bytes, crlfMarker);
  if (crlf >= 0) {
    return crlf + crlfMarker.length;
  }
  throw new Error("Invalid PLY: missing end_header");
}

function indexOfBytes(bytes: Uint8Array, marker: Uint8Array): number {
  for (let i = 0; i <= bytes.length - marker.length; i += 1) {
    let matches = true;
    for (let j = 0; j < marker.length; j += 1) {
      if (bytes[i + j] !== marker[j]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return i;
    }
  }
  return -1;
}

function normalizePlyType(type: string): PlyScalarType {
  switch (type) {
    case "int8":
      return "char";
    case "uint8":
      return "uchar";
    case "int16":
      return "short";
    case "uint16":
      return "ushort";
    case "int32":
      return "int";
    case "uint32":
      return "uint";
    case "float32":
      return "float";
    case "float64":
      return "double";
    case "char":
    case "uchar":
    case "short":
    case "ushort":
    case "int":
    case "uint":
    case "float":
    case "double":
      return type;
    default:
      throw new Error(`Unsupported PLY scalar type: ${type}`);
  }
}

function plyScalarSize(type: PlyScalarType): number {
  switch (type) {
    case "char":
    case "uchar":
      return 1;
    case "short":
    case "ushort":
      return 2;
    case "int":
    case "uint":
    case "float":
      return 4;
    case "double":
      return 8;
  }
}

function readPlyScalar(view: DataView, offset: number, type: PlyScalarType): number {
  switch (type) {
    case "char":
      return view.getInt8(offset);
    case "uchar":
      return view.getUint8(offset);
    case "short":
      return view.getInt16(offset, true);
    case "ushort":
      return view.getUint16(offset, true);
    case "int":
      return view.getInt32(offset, true);
    case "uint":
      return view.getUint32(offset, true);
    case "float":
      return view.getFloat32(offset, true);
    case "double":
      return view.getFloat64(offset, true);
  }
}

function smoothHeightfield(
  heights: Float32Array,
  width: number,
  depth: number,
  passes: number,
): void {
  if (passes <= 0) {
    return;
  }
  const next = new Float32Array(heights.length);
  for (let pass = 0; pass < passes; pass += 1) {
    for (let row = 0; row < depth; row += 1) {
      for (let col = 0; col < width; col += 1) {
        const index = row * width + col;
        let sum = heights[index] * 4;
        let weight = 4;
        for (const [nr, nc] of neighbors(row, col, depth, width)) {
          sum += heights[nr * width + nc];
          weight += 1;
        }
        next[index] = sum / weight;
      }
    }
    heights.set(next);
  }
}

function* neighbors(
  row: number,
  col: number,
  depth: number,
  width: number,
): Generator<[number, number]> {
  if (row > 0) {
    yield [row - 1, col];
  }
  if (row + 1 < depth) {
    yield [row + 1, col];
  }
  if (col > 0) {
    yield [row, col - 1];
  }
  if (col + 1 < width) {
    yield [row, col + 1];
  }
}

function clampHeightfield(heights: Float32Array, elevationScale: number): void {
  for (let i = 0; i < heights.length; i += 1) {
    heights[i] = Math.min(elevationScale, Math.max(0, heights[i]));
  }
}

function sampleHeightfield(
  heights: Float32Array,
  width: number,
  depth: number,
  sizeX: number,
  sizeY: number,
  x: number,
  y: number,
): number {
  const u = clampNumber((0.5 + x / (2 * sizeX)) * (width - 1), 0, width - 1);
  const v = clampNumber((0.5 + y / (2 * sizeY)) * (depth - 1), 0, depth - 1);
  const col0 = Math.floor(u);
  const row0 = Math.floor(v);
  const col1 = Math.min(width - 1, col0 + 1);
  const row1 = Math.min(depth - 1, row0 + 1);
  const tx = u - col0;
  const ty = v - row0;
  const h00 = heights[row0 * width + col0];
  const h10 = heights[row0 * width + col1];
  const h01 = heights[row1 * width + col0];
  const h11 = heights[row1 * width + col1];
  return lerp(lerp(h00, h10, tx), lerp(h01, h11, tx), ty);
}

function sampleStartHeight(
  heights: Float32Array,
  width: number,
  depth: number,
  sizeX: number,
  sizeY: number,
  x: number,
  y: number,
  radius = 0,
): number {
  const sampled = sampleHeightfield(heights, width, depth, sizeX, sizeY, x, y);
  if (radius <= 0) {
    return sampled;
  }

  const u = clampNumber((0.5 + x / (2 * sizeX)) * (width - 1), 0, width - 1);
  const v = clampNumber((0.5 + y / (2 * sizeY)) * (depth - 1), 0, depth - 1);
  const cellSizeX = (2 * sizeX) / Math.max(1, width - 1);
  const cellSizeY = (2 * sizeY) / Math.max(1, depth - 1);
  const cellRadius = Math.ceil(radius / Math.max(1e-6, Math.min(cellSizeX, cellSizeY)));
  const centerCol = Math.round(u);
  const centerRow = Math.round(v);
  let maxHeight = sampled;
  for (
    let row = Math.max(0, centerRow - cellRadius);
    row <= Math.min(depth - 1, centerRow + cellRadius);
    row += 1
  ) {
    for (
      let col = Math.max(0, centerCol - cellRadius);
      col <= Math.min(width - 1, centerCol + cellRadius);
      col += 1
    ) {
      maxHeight = Math.max(maxHeight, heights[row * width + col]);
    }
  }
  return maxHeight;
}

function parseXml(xml: string): XMLDocument {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const parseError = doc.querySelector("parsererror");
  if (parseError) {
    throw new Error(`Failed to parse MuJoCo XML: ${parseError.textContent ?? "unknown error"}`);
  }
  return doc;
}

function ensureAssetElement(doc: XMLDocument): Element {
  const existing = doc.querySelector("asset");
  if (existing) {
    return existing;
  }
  const mujoco = doc.querySelector("mujoco");
  const worldbody = doc.querySelector("worldbody");
  if (!mujoco) {
    throw new Error("MuJoCo XML is missing <mujoco>");
  }
  const asset = doc.createElement("asset");
  mujoco.insertBefore(asset, worldbody ?? mujoco.firstChild);
  return asset;
}

function replaceTerrainBody(doc: XMLDocument, prepared: PreparedHeightfield): void {
  const terrainBody = Array.from(doc.getElementsByTagName("body"))
    .find((body) => body.getAttribute("name") === "terrain");
  if (!terrainBody) {
    throw new Error("MuJoCo XML is missing body name=\"terrain\"");
  }
  while (terrainBody.firstChild) {
    terrainBody.removeChild(terrainBody.firstChild);
  }
  terrainBody.appendChild(doc.createTextNode("\n      "));
  const geom = doc.createElement("geom");
  geom.setAttribute("name", "terrain");
  geom.setAttribute("type", "hfield");
  geom.setAttribute("hfield", "gs_heightfield");
  geom.setAttribute("pos", `${formatNumber(prepared.windowCenterX)} ${formatNumber(prepared.windowCenterY)} 0`);
  geom.setAttribute("material", "groundplane");
  geom.setAttribute("mass", "0");
  geom.setAttribute("friction", "1 0.005 0.0005");
  geom.setAttribute("rgba", "0.22 0.42 0.32 1");
  terrainBody.appendChild(geom);
  terrainBody.appendChild(doc.createTextNode("\n    "));
}

function updateInitKeyframe(doc: XMLDocument, prepared: PreparedHeightfield): void {
  const key = Array.from(doc.getElementsByTagName("key"))
    .find((item) => item.getAttribute("name") === "init_state");
  const qpos = key?.getAttribute("qpos");
  if (!key || !qpos) {
    return;
  }
  const values = qpos.trim().split(/\s+/).map(Number);
  if (values.length < 3) {
    return;
  }
  values[0] = prepared.config.startX;
  values[1] = prepared.config.startY;
  values[2] = prepared.startHeight + prepared.config.robotBaseHeight;

  // Apply custom spawn yaw as a quaternion if specified
  if (prepared.customSpawn && prepared.customSpawn.length >= 3 && values.length >= 7) {
    const yaw = prepared.customSpawn[2];
    const half = yaw * 0.5;
    const qw = Math.cos(half);
    const qz = Math.sin(half);
    values[3] = qw; // qw
    values[4] = 0;  // qx
    values[5] = 0;  // qy
    values[6] = qz; // qz
  }

  key.setAttribute("qpos", values.map(formatNumber).join(" "));
}

function heightfieldElevationString(prepared: PreparedHeightfield): string {
  const values = new Array<string>(prepared.heights.length);
  let target = 0;
  // MuJoCo reverses XML hfield rows on load; write top-to-bottom to preserve local Y.
  for (let row = prepared.config.nrow - 1; row >= 0; row -= 1) {
    for (let col = 0; col < prepared.config.ncol; col += 1) {
      const source = row * prepared.config.ncol + col;
      values[target] = formatNumber(prepared.heights[source] / prepared.config.elevationScale);
      target += 1;
    }
  }
  return values.join(" ");
}

function gridX(config: BaseHeightfieldConfig, col: number): number {
  return -config.sizeX + (2 * config.sizeX * col) / (config.ncol - 1);
}

function gridY(config: BaseHeightfieldConfig, row: number): number {
  return -config.sizeY + (2 * config.sizeY * row) / (config.nrow - 1);
}

function smoothStep(t: number): number {
  return t * t * (3 - 2 * t);
}

function normalizeQuat(quat: [number, number, number, number]): [number, number, number, number] {
  const norm = Math.hypot(quat[0], quat[1], quat[2], quat[3]) || 1;
  return [quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp01(value: number): number {
  return clampNumber(value, 0, 1);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampInteger(value: number, min: number, max: number): number {
  return Math.trunc(clampNumber(value, min, max));
}

function roundUp(value: number, step: number): number {
  return Math.ceil(value / step) * step;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return value.toFixed(6).replace(/\.?0+$/, "");
}
