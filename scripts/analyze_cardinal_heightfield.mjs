#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const DEFAULT_SOURCE = "public/envs/go1_gaussian/splats/cardinal/cardinal_towers.splat";

const DEFAULT_CONFIG = {
  baseNrow: 97,
  baseNcol: 193,
  baseSizeX: 6,
  baseSizeY: 6,
  elevationScale: 9.5,
  sourceBoundsPadding: 1,
  sourceTargetCellSize: 0.125,
  sourceMaxRows: 160,
  sourceMaxCols: 256,
  supportChunkSize: 8,
  supportChunkResolution: 96,
  supportChunkPadding: 2,
  supportFillIterations: 8,
  supportFillMaxHeightRange: 0.45,
};

const DEFAULT_OPTIONS = {
  source: DEFAULT_SOURCE,
  sigmaRadius: 2.0,
  minOpacity: 0.15,
  driftThreshold: 0.25,
  elevatedThreshold: 0.35,
  fillMode: "nearby",
  csv: "",
};

const CARDINAL_TOWERS = [
  { label: "east", x: 4, z: 0, height: 0.75 },
  { label: "north", x: 0, z: 4, height: 2.5 },
  { label: "west", x: -4, z: 0, height: 5 },
  { label: "south", x: 0, z: -4, height: 9 },
];

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const sourcePath = resolve(options.source);
  const splats = parseSplat(await readFile(sourcePath));
  const bounds = computeBounds(splats, options.sigmaRadius);
  const config = resolveConfig(DEFAULT_CONFIG, bounds);
  const tileCoords = supportColumnCoords(splats, config);
  const tiles = new Map();

  for (const coord of tileCoords) {
    const tile = createSupportTile(splats, coord[0], coord[1], config, options);
    tiles.set(tileKey(coord[0], coord[1]), tile);
  }

  const appHeights = sampleHeightfield(tiles, config, sampleModeForFillMode(options.fillMode));
  const noFallbackHeights = sampleHeightfield(tiles, config, "no-empty-fallback");
  const originalOnlyHeights = sampleHeightfield(tiles, config, "original-only");
  const cells = analyzeCells(appHeights, noFallbackHeights, originalOnlyHeights, config, options);
  const tileStats = [...tiles.entries()].map(([key, tile]) => summarizeTile(key, tile));
  const towerStats = CARDINAL_TOWERS.map((tower) =>
    summarizeTower(tower, appHeights, noFallbackHeights, originalOnlyHeights, config, options),
  );

  printReport({
    sourcePath,
    splats,
    bounds,
    config,
    options,
    tileStats,
    cells,
    towerStats,
  });

  if (options.csv) {
    await writeCsv(resolve(options.csv), cells.rows, config);
  }
}

function parseArgs(args) {
  const options = { ...DEFAULT_OPTIONS };
  for (const arg of args) {
    const [name, rawValue] = arg.replace(/^--/, "").split("=");
    const value = rawValue ?? "";
    switch (name) {
      case "source":
        options.source = value;
        break;
      case "sigma-radius":
        options.sigmaRadius = numberOption(name, value);
        break;
      case "min-opacity":
        options.minOpacity = numberOption(name, value);
        break;
      case "drift-threshold":
        options.driftThreshold = numberOption(name, value);
        break;
      case "elevated-threshold":
        options.elevatedThreshold = numberOption(name, value);
        break;
      case "fill-iterations":
        DEFAULT_CONFIG.supportFillIterations = numberOption(name, value);
        break;
      case "fill-max-height-range":
        DEFAULT_CONFIG.supportFillMaxHeightRange = numberOption(name, value);
        break;
      case "fill-mode":
        if (!["nearby", "fallback", "min"].includes(value)) {
          throw new Error(`Invalid --fill-mode=${value}; expected nearby, fallback, or min`);
        }
        options.fillMode = value;
        break;
      case "csv":
        options.csv = value;
        break;
      case "help":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }
  return options;
}

function printHelp() {
  console.log(`Usage: node scripts/analyze_cardinal_heightfield.mjs [options]

Options:
  --source=PATH              .splat source to analyze
  --sigma-radius=N          Horizontal Gaussian influence radius in splat sigmas
  --min-opacity=N           Minimum opacity in normalized 0..1 units
  --drift-threshold=N       Height delta reported as fallback drift
  --elevated-threshold=N    Height treated as elevated terrain
  --fill-iterations=N       Hole-fill mip levels to simulate
  --fill-max-height-range=N Maximum local range allowed while filling holes
  --fill-mode=MODE          App unresolved-hole mode: nearby, fallback, or min
  --csv=PATH                Write per-cell diagnostics as CSV
`);
}

function numberOption(name, value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid numeric option --${name}=${value}`);
  }
  return parsed;
}

function parseSplat(bytes) {
  if (bytes.byteLength % 32 !== 0) {
    throw new Error(`Invalid .splat byte length: ${bytes.byteLength}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const splats = [];
  for (let offset = 0; offset < bytes.byteLength; offset += 32) {
    splats.push({
      x: view.getFloat32(offset, true),
      y: view.getFloat32(offset + 4, true),
      z: view.getFloat32(offset + 8, true),
      scaleX: Math.max(1e-6, view.getFloat32(offset + 12, true)),
      scaleY: Math.max(1e-6, view.getFloat32(offset + 16, true)),
      scaleZ: Math.max(1e-6, view.getFloat32(offset + 20, true)),
      opacity: bytes[offset + 27] / 255,
    });
  }
  return splats;
}

function computeBounds(splats, sigmaRadius) {
  const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (const splat of splats) {
    min[0] = Math.min(min[0], splat.x - sigmaRadius * splat.scaleX);
    min[1] = Math.min(min[1], splat.y - sigmaRadius * splat.scaleY);
    min[2] = Math.min(min[2], splat.z - sigmaRadius * splat.scaleZ);
    max[0] = Math.max(max[0], splat.x + sigmaRadius * splat.scaleX);
    max[1] = Math.max(max[1], splat.y + sigmaRadius * splat.scaleY);
    max[2] = Math.max(max[2], splat.z + sigmaRadius * splat.scaleZ);
  }
  return { min, max };
}

function resolveConfig(base, bounds) {
  const sizeX = roundUp(
    Math.max(base.baseSizeX, Math.abs(bounds.min[0]) + base.sourceBoundsPadding, Math.abs(bounds.max[0]) + base.sourceBoundsPadding),
    0.5,
  );
  const sizeY = roundUp(
    Math.max(base.baseSizeY, Math.abs(bounds.min[2]) + base.sourceBoundsPadding, Math.abs(bounds.max[2]) + base.sourceBoundsPadding),
    0.5,
  );
  const nrow = clampInteger(
    Math.ceil((2 * sizeY) / base.sourceTargetCellSize) + 1,
    base.baseNrow,
    Math.max(base.baseNrow, base.sourceMaxRows),
  );
  const ncol = clampInteger(
    Math.ceil((2 * sizeX) / base.sourceTargetCellSize) + 1,
    base.baseNcol,
    Math.max(base.baseNcol, base.sourceMaxCols),
  );
  return {
    ...base,
    sizeX,
    sizeY,
    nrow,
    ncol,
  };
}

function supportColumnCoords(splats, config) {
  const coords = new Set();
  for (const splat of splats) {
    const minCx = Math.floor((splat.x - config.supportChunkPadding) / config.supportChunkSize);
    const maxCx = Math.floor((splat.x + config.supportChunkPadding) / config.supportChunkSize);
    const minCz = Math.floor((splat.z - config.supportChunkPadding) / config.supportChunkSize);
    const maxCz = Math.floor((splat.z + config.supportChunkPadding) / config.supportChunkSize);
    for (let cx = minCx; cx <= maxCx; cx += 1) {
      for (let cz = minCz; cz <= maxCz; cz += 1) {
        coords.add(tileKey(cx, cz));
      }
    }
  }
  return [...coords].map((key) => key.split(",").map(Number));
}

function createSupportTile(splats, cx, cz, config, options) {
  const width = config.supportChunkResolution;
  const depth = config.supportChunkResolution;
  const voxelSize = config.supportChunkSize / config.supportChunkResolution;
  const height = new Float32Array(width * depth);
  const state = new Uint8Array(width * depth);
  height.fill(Number.NaN);

  const tileMinX = cx * config.supportChunkSize;
  const tileMinZ = cz * config.supportChunkSize;
  const tileSplats = splats.filter((splat) => {
    const minCx = Math.floor((splat.x - config.supportChunkPadding) / config.supportChunkSize);
    const maxCx = Math.floor((splat.x + config.supportChunkPadding) / config.supportChunkSize);
    const minCz = Math.floor((splat.z - config.supportChunkPadding) / config.supportChunkSize);
    const maxCz = Math.floor((splat.z + config.supportChunkPadding) / config.supportChunkSize);
    return cx >= minCx && cx <= maxCx && cz >= minCz && cz <= maxCz && splat.opacity >= options.minOpacity;
  });

  for (let row = 0; row < depth; row += 1) {
    const z = tileMinZ + (row + 0.5) * voxelSize;
    for (let col = 0; col < width; col += 1) {
      const x = tileMinX + (col + 0.5) * voxelSize;
      let top = Number.NEGATIVE_INFINITY;
      for (const splat of tileSplats) {
        const radius = options.sigmaRadius * Math.max(splat.scaleX, splat.scaleZ);
        const dx = x - splat.x;
        const dz = z - splat.z;
        if (dx * dx + dz * dz > radius * radius) {
          continue;
        }
        top = Math.max(top, splat.y + options.sigmaRadius * splat.scaleY);
      }
      if (Number.isFinite(top)) {
        const index = col + row * width;
        height[index] = top;
        state[index] = 1;
      }
    }
  }

  const tile = {
    height,
    state,
    width,
    depth,
    fallbackHeight: Number.NaN,
  };
  fillHeightFieldHoles(tile, config.supportFillIterations, config.supportFillMaxHeightRange);
  tile.fallbackHeight = getHeightFieldFallback(tile);
  return tile;
}

function sampleHeightfield(tiles, config, mode) {
  const heights = new Float32Array(config.nrow * config.ncol);
  for (let row = 0; row < config.nrow; row += 1) {
    for (let col = 0; col < config.ncol; col += 1) {
      const index = row * config.ncol + col;
      heights[index] = sampleSupportHeight(
        tiles,
        gridX(config, col),
        gridY(config, row),
        config.supportChunkSize,
        config.supportChunkResolution,
        mode,
      );
    }
  }
  return heights;
}

function sampleModeForFillMode(fillMode) {
  return fillMode === "nearby" ? "no-empty-fallback" : "app-fallback";
}

function sampleSupportHeight(tiles, x, z, chunkSize, chunkResolution, mode) {
  const voxelSize = chunkSize / chunkResolution;
  const gx = x / voxelSize - 0.5;
  const gz = z / voxelSize - 0.5;
  const gx0 = Math.floor(gx);
  const gz0 = Math.floor(gz);
  const tx = gx - gx0;
  const tz = gz - gz0;
  let sumHeight = 0;
  let sumWeight = 0;

  for (let dz = 0; dz <= 1; dz += 1) {
    for (let dx = 0; dx <= 1; dx += 1) {
      const sampleX = gx0 + dx;
      const sampleZ = gz0 + dz;
      const tileCx = Math.floor(sampleX / chunkResolution);
      const tileCz = Math.floor(sampleZ / chunkResolution);
      const tile = tiles.get(tileKey(tileCx, tileCz));
      if (!tile) {
        continue;
      }
      const vx = sampleX - tileCx * chunkResolution;
      const vz = sampleZ - tileCz * chunkResolution;
      if (vx < 0 || vx >= tile.width || vz < 0 || vz >= tile.depth) {
        continue;
      }
      const height = resolveSupportHeight(tile, vx + vz * tile.width, mode);
      if (height === undefined) {
        continue;
      }
      const weightX = dx === 0 ? 1 - tx : tx;
      const weightZ = dz === 0 ? 1 - tz : tz;
      const weight = weightX * weightZ;
      sumHeight += height * weight;
      sumWeight += weight;
    }
  }

  return sumWeight > 0.0001 ? sumHeight / sumWeight : 0;
}

function resolveSupportHeight(field, index, mode) {
  const fallbackHeight = Number.isFinite(field.fallbackHeight) ? field.fallbackHeight : undefined;
  if (mode === "original-only" && field.state[index] !== 1) {
    return undefined;
  }
  if (field.state[index] === 0) {
    return mode === "app-fallback" ? fallbackHeight : undefined;
  }
  const height = field.height[index];
  if (Number.isFinite(height)) {
    return height;
  }
  return mode === "app-fallback" ? fallbackHeight : undefined;
}

function analyzeCells(appHeights, noFallbackHeights, originalOnlyHeights, config, options) {
  const rows = [];
  let elevatedApp = 0;
  let elevatedNoFallback = 0;
  let driftCells = 0;
  let maxDrift = Number.NEGATIVE_INFINITY;
  let maxHeight = Number.NEGATIVE_INFINITY;

  for (let row = 0; row < config.nrow; row += 1) {
    for (let col = 0; col < config.ncol; col += 1) {
      const index = row * config.ncol + col;
      const app = appHeights[index];
      const noFallback = noFallbackHeights[index];
      const originalOnly = originalOnlyHeights[index];
      const drift = app - noFallback;
      const x = gridX(config, col);
      const z = gridY(config, row);
      const tower = nearestTower(x, z);
      if (app >= options.elevatedThreshold) {
        elevatedApp += 1;
      }
      if (noFallback >= options.elevatedThreshold) {
        elevatedNoFallback += 1;
      }
      if (drift >= options.driftThreshold) {
        driftCells += 1;
      }
      maxDrift = Math.max(maxDrift, drift);
      maxHeight = Math.max(maxHeight, app);
      rows.push({
        row,
        col,
        x,
        z,
        app,
        noFallback,
        originalOnly,
        drift,
        nearestTower: tower.label,
        towerDistance: tower.distance,
      });
    }
  }

  rows.sort((a, b) => b.drift - a.drift || b.app - a.app);
  return {
    rows,
    elevatedApp,
    elevatedNoFallback,
    driftCells,
    maxDrift,
    maxHeight,
    totalCells: config.nrow * config.ncol,
  };
}

function summarizeTile(key, tile) {
  let original = 0;
  let filled = 0;
  let empty = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < tile.state.length; i += 1) {
    if (tile.state[i] === 0) {
      empty += 1;
      continue;
    }
    if (tile.state[i] === 1) {
      original += 1;
    } else {
      filled += 1;
    }
    const height = tile.height[i];
    if (!Number.isFinite(height)) {
      continue;
    }
    min = Math.min(min, height);
    max = Math.max(max, height);
    sum += height;
    count += 1;
  }
  return {
    key,
    original,
    filled,
    empty,
    fallbackHeight: tile.fallbackHeight,
    min,
    max,
    mean: count > 0 ? sum / count : Number.NaN,
  };
}

function summarizeTower(tower, appHeights, noFallbackHeights, originalOnlyHeights, config, options) {
  const rings = [
    { label: "core <=0.65m", min: 0, max: 0.65 },
    { label: "near 0.65-1.5m", min: 0.65, max: 1.5 },
    { label: "outer 1.5-2.5m", min: 1.5, max: 2.5 },
  ];
  return {
    label: tower.label,
    height: tower.height,
    rings: rings.map((ring) => {
      let count = 0;
      let appSum = 0;
      let noFallbackSum = 0;
      let originalSum = 0;
      let driftSum = 0;
      let maxDrift = Number.NEGATIVE_INFINITY;
      let driftCells = 0;
      for (let row = 0; row < config.nrow; row += 1) {
        for (let col = 0; col < config.ncol; col += 1) {
          const x = gridX(config, col);
          const z = gridY(config, row);
          const distance = Math.hypot(x - tower.x, z - tower.z);
          if (distance < ring.min || distance >= ring.max) {
            continue;
          }
          const index = row * config.ncol + col;
          const app = appHeights[index];
          const noFallback = noFallbackHeights[index];
          const original = originalOnlyHeights[index];
          const drift = app - noFallback;
          count += 1;
          appSum += app;
          noFallbackSum += noFallback;
          originalSum += original;
          driftSum += drift;
          maxDrift = Math.max(maxDrift, drift);
          if (drift >= options.driftThreshold) {
            driftCells += 1;
          }
        }
      }
      return {
        ...ring,
        count,
        appMean: count > 0 ? appSum / count : Number.NaN,
        noFallbackMean: count > 0 ? noFallbackSum / count : Number.NaN,
        originalMean: count > 0 ? originalSum / count : Number.NaN,
        driftMean: count > 0 ? driftSum / count : Number.NaN,
        maxDrift,
        driftCells,
      };
    }),
  };
}

function printReport(report) {
  const { sourcePath, splats, bounds, config, options, tileStats, cells, towerStats } = report;
  console.log("Cardinal heightfield drift analysis");
  console.log(`source: ${sourcePath}`);
  console.log(`splats: ${splats.length}`);
  console.log(`bounds: x ${fmt(bounds.min[0])}..${fmt(bounds.max[0])}, y ${fmt(bounds.min[1])}..${fmt(bounds.max[1])}, z ${fmt(bounds.min[2])}..${fmt(bounds.max[2])}`);
  console.log(`hfield: ${config.ncol} x ${config.nrow}, size ${fmt(config.sizeX)} x ${fmt(config.sizeY)}, tile ${config.supportChunkResolution}^2`);
  console.log(`model: sigmaRadius=${options.sigmaRadius}, minOpacity=${options.minOpacity}, fillMode=${options.fillMode}, driftThreshold=${options.driftThreshold}`);
  console.log("");

  console.log("Support tile fallback heights");
  for (const tile of tileStats.sort((a, b) => a.key.localeCompare(b.key))) {
    console.log(
      `  ${tile.key}: fallback=${fmt(tile.fallbackHeight)} original=${tile.original} filled=${tile.filled} empty=${tile.empty} range=${fmt(tile.min)}..${fmt(tile.max)}`,
    );
  }
  console.log("");

  const pct = (value) => `${fmt((100 * value) / cells.totalCells)}%`;
  console.log("Window summary");
  console.log(`  elevated cells with app fallback: ${cells.elevatedApp}/${cells.totalCells} (${pct(cells.elevatedApp)})`);
  console.log(`  elevated cells without empty fallback: ${cells.elevatedNoFallback}/${cells.totalCells} (${pct(cells.elevatedNoFallback)})`);
  console.log(`  cells where fallback adds >= ${options.driftThreshold}m: ${cells.driftCells}/${cells.totalCells} (${pct(cells.driftCells)})`);
  console.log(`  max fallback drift: ${fmt(cells.maxDrift)}m`);
  console.log("");

  console.log("Tower ring means");
  for (const tower of towerStats) {
    console.log(`  ${tower.label} tower (${tower.height}m)`);
    for (const ring of tower.rings) {
      console.log(
        `    ${ring.label}: app=${fmt(ring.appMean)} noFallback=${fmt(ring.noFallbackMean)} originalOnly=${fmt(ring.originalMean)} drift=${fmt(ring.driftMean)} maxDrift=${fmt(ring.maxDrift)} driftCells=${ring.driftCells}/${ring.count}`,
      );
    }
  }
  console.log("");

  console.log("Top fallback-drift cells");
  for (const row of cells.rows.slice(0, 20)) {
    console.log(
      `  x=${fmt(row.x)} z=${fmt(row.z)} app=${fmt(row.app)} noFallback=${fmt(row.noFallback)} originalOnly=${fmt(row.originalOnly)} drift=${fmt(row.drift)} nearest=${row.nearestTower}@${fmt(row.towerDistance)}m`,
    );
  }
}

async function writeCsv(path, rows, config) {
  const header = "row,col,x,z,app_height,no_empty_fallback_height,original_only_height,fallback_drift,nearest_tower,tower_distance\n";
  const body = rows
    .slice()
    .sort((a, b) => a.row - b.row || a.col - b.col)
    .map((row) => [
      row.row,
      row.col,
      row.x,
      row.z,
      row.app,
      row.noFallback,
      row.originalOnly,
      row.drift,
      row.nearestTower,
      row.towerDistance,
    ].join(","))
    .join("\n");
  await writeFile(path, header + body + "\n");
  console.log("");
  console.log(`wrote CSV: ${path} (${config.ncol * config.nrow} cells)`);
}

function fillHeightFieldHoles(field, iterations, maxHeightRange) {
  const maxLevels = Math.max(0, Math.floor(iterations));
  if (maxLevels === 0) {
    return;
  }
  fillHeightFieldHolesFromFourNeighbors(field, maxHeightRange);
  if (maxLevels === 1) {
    return;
  }
  const mipmap = buildHeightMipmap(field, maxLevels);
  if (mipmap.length <= 1) {
    return;
  }
  const { width, depth } = field;
  for (let z = 0; z < depth; z += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + z * width;
      if (field.state[index] !== 0) {
        continue;
      }
      const height = sampleHeightMipmap(mipmap, x, z, maxHeightRange);
      if (height === undefined) {
        continue;
      }
      field.height[index] = height;
      field.state[index] = 2;
    }
  }
}

function fillHeightFieldHolesFromFourNeighbors(field, maxHeightRange) {
  const { width, depth } = field;
  const nextHeight = new Float32Array(field.height.length);
  const nextState = new Uint8Array(field.state.length);
  nextHeight.set(field.height);
  nextState.set(field.state);
  for (let z = 0; z < depth; z += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + z * width;
      if (field.state[index] !== 0) {
        continue;
      }
      const height = sampleFourNeighborHeight(field, x, z, maxHeightRange);
      if (height === undefined) {
        continue;
      }
      nextHeight[index] = height;
      nextState[index] = 2;
    }
  }
  field.height.set(nextHeight);
  field.state.set(nextState);
}

function sampleFourNeighborHeight(field, x, z, maxHeightRange) {
  const offsets = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  let sum = 0;
  let count = 0;
  let minHeight = Number.POSITIVE_INFINITY;
  let maxHeight = Number.NEGATIVE_INFINITY;
  for (const [dx, dz] of offsets) {
    const nx = x + dx;
    const nz = z + dz;
    if (nx < 0 || nx >= field.width || nz < 0 || nz >= field.depth) {
      continue;
    }
    const index = nx + nz * field.width;
    if (field.state[index] === 0) {
      continue;
    }
    const height = field.height[index];
    if (!Number.isFinite(height)) {
      continue;
    }
    sum += height;
    count += 1;
    minHeight = Math.min(minHeight, height);
    maxHeight = Math.max(maxHeight, height);
  }
  if (count === 0 || maxHeight - minHeight > maxHeightRange) {
    return undefined;
  }
  return sum / count;
}

function getHeightFieldFallback(field) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < field.height.length; i += 1) {
    if (field.state[i] === 0) {
      continue;
    }
    const height = field.height[i];
    if (!Number.isFinite(height)) {
      continue;
    }
    sum += height;
    count += 1;
  }
  return count > 0 ? sum / count : Number.NaN;
}

function buildHeightMipmap(field, maxLevels) {
  const levels = [createBaseHeightMipmapLevel(field)];
  for (let levelIndex = 1; levelIndex < maxLevels; levelIndex += 1) {
    const previous = levels[levelIndex - 1];
    if (previous.width === 1 && previous.depth === 1) {
      break;
    }
    const width = Math.max(1, Math.ceil(previous.width / 2));
    const depth = Math.max(1, Math.ceil(previous.depth / 2));
    const next = createEmptyHeightMipmapLevel(width, depth);
    for (let z = 0; z < depth; z += 1) {
      for (let x = 0; x < width; x += 1) {
        const targetIndex = x + z * width;
        for (let dz = 0; dz < 2; dz += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const px = x * 2 + dx;
            const pz = z * 2 + dz;
            if (px >= previous.width || pz >= previous.depth) {
              continue;
            }
            const sourceIndex = px + pz * previous.width;
            const count = previous.count[sourceIndex];
            if (count === 0) {
              continue;
            }
            next.sum[targetIndex] += previous.sum[sourceIndex];
            next.count[targetIndex] += count;
            next.min[targetIndex] = Math.min(next.min[targetIndex], previous.min[sourceIndex]);
            next.max[targetIndex] = Math.max(next.max[targetIndex], previous.max[sourceIndex]);
          }
        }
      }
    }
    levels.push(next);
  }
  return levels;
}

function createBaseHeightMipmapLevel(field) {
  const level = createEmptyHeightMipmapLevel(field.width, field.depth);
  for (let i = 0; i < field.height.length; i += 1) {
    if (field.state[i] === 0) {
      continue;
    }
    const height = field.height[i];
    if (!Number.isFinite(height)) {
      continue;
    }
    level.sum[i] = height;
    level.count[i] = 1;
    level.min[i] = height;
    level.max[i] = height;
  }
  return level;
}

function createEmptyHeightMipmapLevel(width, depth) {
  const length = width * depth;
  const min = new Float32Array(length);
  const max = new Float32Array(length);
  min.fill(Number.POSITIVE_INFINITY);
  max.fill(Number.NEGATIVE_INFINITY);
  return {
    width,
    depth,
    sum: new Float32Array(length),
    count: new Uint32Array(length),
    min,
    max,
  };
}

function sampleHeightMipmap(levels, x, z, maxHeightRange) {
  for (let levelIndex = 1; levelIndex < levels.length; levelIndex += 1) {
    const level = levels[levelIndex];
    const scale = 2 ** levelIndex;
    const coarseX = Math.floor(x / scale);
    const coarseZ = Math.floor(z / scale);
    const localX = x - coarseX * scale;
    const localZ = z - coarseZ * scale;
    const startX = getMipmapWindowStart(coarseX, localX, scale, level.width);
    const startZ = getMipmapWindowStart(coarseZ, localZ, scale, level.depth);
    let sum = 0;
    let count = 0;
    let minHeight = Number.POSITIVE_INFINITY;
    let maxHeight = Number.NEGATIVE_INFINITY;
    for (let dz = 0; dz < Math.min(2, level.depth); dz += 1) {
      for (let dx = 0; dx < Math.min(2, level.width); dx += 1) {
        const sampleIndex = startX + dx + (startZ + dz) * level.width;
        const sampleCount = level.count[sampleIndex];
        if (sampleCount === 0) {
          continue;
        }
        sum += level.sum[sampleIndex];
        count += sampleCount;
        minHeight = Math.min(minHeight, level.min[sampleIndex]);
        maxHeight = Math.max(maxHeight, level.max[sampleIndex]);
      }
    }
    if (count === 0 || maxHeight - minHeight > maxHeightRange) {
      continue;
    }
    return sum / count;
  }
  return undefined;
}

function getMipmapWindowStart(coarse, local, scale, size) {
  if (size <= 1) {
    return 0;
  }
  const start = local < scale / 2 ? coarse - 1 : coarse;
  return Math.max(0, Math.min(size - 2, start));
}

function nearestTower(x, z) {
  let best = CARDINAL_TOWERS[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const tower of CARDINAL_TOWERS) {
    const distance = Math.hypot(x - tower.x, z - tower.z);
    if (distance < bestDistance) {
      best = tower;
      bestDistance = distance;
    }
  }
  return { ...best, distance: bestDistance };
}

function gridX(config, col) {
  return -config.sizeX + (2 * config.sizeX * col) / (config.ncol - 1);
}

function gridY(config, row) {
  return -config.sizeY + (2 * config.sizeY * row) / (config.nrow - 1);
}

function tileKey(cx, cz) {
  return `${cx},${cz}`;
}

function roundUp(value, step) {
  return Math.ceil(value / step) * step;
}

function clampInteger(value, min, max) {
  return Math.max(min, Math.min(max, Math.floor(value)));
}

function fmt(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}
