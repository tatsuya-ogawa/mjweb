type MujocoModel = any;

export type Vec3 = [number, number, number];

export interface PlannedHeightfieldPath {
  points: Vec3[];
  expanded: number;
  totalCost: number;
}

export interface HeightfieldPathOptions {
  maxSlope?: number;
  maxStepHeight?: number;
  footprintRadius?: number;
  waypointSpacing?: number;
  slopeCostWeight?: number;
  roughnessCostWeight?: number;
  maxNearestSearchMeters?: number;
  maxExpanded?: number;
}

export interface HeightfieldPathGrid {
  nrow: number;
  ncol: number;
  adr: number;
  sizeX: number;
  sizeY: number;
  sizeZ: number;
  originX: number;
  originY: number;
  cellSizeX: number;
  cellSizeY: number;
  data: ArrayLike<number>;
}

interface PlannerConfig {
  maxSlope: number;
  maxStepHeight: number;
  footprintRadius: number;
  waypointSpacing: number;
  slopeCostWeight: number;
  roughnessCostWeight: number;
  maxNearestSearchMeters: number;
  maxExpanded: number;
}

const NEIGHBOR_COL = [1, 1, 0, -1, -1, -1, 0, 1];
const NEIGHBOR_ROW = [0, 1, 1, 1, 0, -1, -1, -1];

export function planHeightfieldPath(
  model: MujocoModel,
  start: readonly [number, number],
  goal: readonly [number, number],
  options: HeightfieldPathOptions = {},
): PlannedHeightfieldPath | null {
  const grid = extractPrimaryHeightfield(model);
  if (!grid) {
    return null;
  }

  return planHeightfieldGridPath(grid, start, goal, options);
}

export function planHeightfieldGridPath(
  grid: HeightfieldPathGrid,
  start: readonly [number, number],
  goal: readonly [number, number],
  options: HeightfieldPathOptions = {},
): PlannedHeightfieldPath | null {
  const total = grid.nrow * grid.ncol;
  const config = resolveConfig(options, total);
  const walkable = new Int8Array(total);
  walkable.fill(-1);
  const roughness = new Float32Array(total);
  roughness.fill(Number.NaN);

  const startIndex = nearestWalkableIndex(
    grid,
    worldToIndex(grid, start[0], start[1]),
    config,
    walkable,
    roughness,
  );
  const goalIndex = nearestWalkableIndex(
    grid,
    worldToIndex(grid, goal[0], goal[1]),
    config,
    walkable,
    roughness,
  );
  if (startIndex < 0 || goalIndex < 0) {
    return null;
  }

  const gScore = new Float32Array(total);
  gScore.fill(Number.POSITIVE_INFINITY);
  const cameFrom = new Int32Array(total);
  cameFrom.fill(-1);
  const closed = new Uint8Array(total);
  const open = new BinaryHeap();

  gScore[startIndex] = 0;
  open.push(startIndex, heuristic(grid, startIndex, goalIndex));

  let expanded = 0;
  while (!open.isEmpty() && expanded < config.maxExpanded) {
    const current = open.pop();
    if (current < 0 || closed[current]) {
      continue;
    }
    if (current === goalIndex) {
      const indices = reconstructPath(cameFrom, current);
      return {
        points: indicesToWaypoints(grid, indices, config.waypointSpacing),
        expanded,
        totalCost: gScore[current],
      };
    }
    closed[current] = 1;
    expanded += 1;

    const col = current % grid.ncol;
    const row = Math.floor(current / grid.ncol);
    for (let direction = 0; direction < NEIGHBOR_COL.length; direction += 1) {
      const nextCol = col + NEIGHBOR_COL[direction];
      const nextRow = row + NEIGHBOR_ROW[direction];
      if (nextCol < 0 || nextCol >= grid.ncol || nextRow < 0 || nextRow >= grid.nrow) {
        continue;
      }
      const next = nextRow * grid.ncol + nextCol;
      if (closed[next] || !isWalkable(grid, next, config, walkable, roughness)) {
        continue;
      }
      const moveCost = edgeCost(grid, current, next, config, roughness);
      if (!Number.isFinite(moveCost)) {
        continue;
      }
      const tentative = gScore[current] + moveCost;
      if (tentative >= gScore[next]) {
        continue;
      }
      cameFrom[next] = current;
      gScore[next] = tentative;
      open.push(next, tentative + heuristic(grid, next, goalIndex));
    }
  }

  return null;
}

function extractPrimaryHeightfield(model: MujocoModel): HeightfieldPathGrid | null {
  const count = model.nhfield ?? Math.floor((model.hfield_size?.length ?? 0) / 4);
  for (let hfieldId = 0; hfieldId < count; hfieldId += 1) {
    const nrow = model.hfield_nrow?.[hfieldId] ?? 0;
    const ncol = model.hfield_ncol?.[hfieldId] ?? 0;
    const adr = model.hfield_adr?.[hfieldId] ?? -1;
    const sizeBase = hfieldId * 4;
    const sizeX = model.hfield_size?.[sizeBase] ?? 0;
    const sizeY = model.hfield_size?.[sizeBase + 1] ?? 0;
    const sizeZ = model.hfield_size?.[sizeBase + 2] ?? 0;
    if (nrow >= 2 && ncol >= 2 && adr >= 0 && sizeX > 0 && sizeY > 0 && sizeZ > 0) {
      const origin = hfieldOrigin(model, hfieldId);
      return {
        nrow,
        ncol,
        adr,
        sizeX,
        sizeY,
        sizeZ,
        originX: origin[0],
        originY: origin[1],
        cellSizeX: (2 * sizeX) / (ncol - 1),
        cellSizeY: (2 * sizeY) / (nrow - 1),
        data: model.hfield_data,
      };
    }
  }
  return null;
}

function hfieldOrigin(model: MujocoModel, hfieldId: number): [number, number] {
  const geomCount = model.ngeom ?? Math.floor((model.geom_dataid?.length ?? 0));
  for (let geomId = 0; geomId < geomCount; geomId += 1) {
    if (model.geom_dataid?.[geomId] !== hfieldId) {
      continue;
    }
    if (model.geom_type && model.geom_type[geomId] !== 9) {
      continue;
    }
    const base = geomId * 3;
    return [
      model.geom_pos?.[base] ?? 0,
      model.geom_pos?.[base + 1] ?? 0,
    ];
  }
  return [0, 0];
}

function resolveConfig(options: HeightfieldPathOptions, totalCells: number): PlannerConfig {
  return {
    maxSlope: options.maxSlope ?? 0.55,
    maxStepHeight: options.maxStepHeight ?? 0.22,
    footprintRadius: options.footprintRadius ?? 0.26,
    waypointSpacing: options.waypointSpacing ?? 0.45,
    slopeCostWeight: options.slopeCostWeight ?? 5.0,
    roughnessCostWeight: options.roughnessCostWeight ?? 3.0,
    maxNearestSearchMeters: options.maxNearestSearchMeters ?? 3.0,
    maxExpanded: Math.min(options.maxExpanded ?? totalCells, totalCells),
  };
}

function worldToIndex(grid: HeightfieldPathGrid, x: number, y: number): number {
  const localX = x - grid.originX;
  const localY = y - grid.originY;
  const col = Math.round(clamp((localX + grid.sizeX) / (2 * grid.sizeX), 0, 1) * (grid.ncol - 1));
  const row = Math.round(clamp((localY + grid.sizeY) / (2 * grid.sizeY), 0, 1) * (grid.nrow - 1));
  return row * grid.ncol + col;
}

function indexToWorld(grid: HeightfieldPathGrid, index: number): Vec3 {
  const col = index % grid.ncol;
  const row = Math.floor(index / grid.ncol);
  return [
    grid.originX - grid.sizeX + col * grid.cellSizeX,
    grid.originY - grid.sizeY + row * grid.cellSizeY,
    heightAt(grid, index),
  ];
}

function heightAt(grid: HeightfieldPathGrid, index: number): number {
  return grid.data[grid.adr + index] * grid.sizeZ;
}

function nearestWalkableIndex(
  grid: HeightfieldPathGrid,
  origin: number,
  config: PlannerConfig,
  walkable: Int8Array,
  roughness: Float32Array,
): number {
  if (isWalkable(grid, origin, config, walkable, roughness)) {
    return origin;
  }

  const originCol = origin % grid.ncol;
  const originRow = Math.floor(origin / grid.ncol);
  const maxRadius = Math.ceil(
    config.maxNearestSearchMeters / Math.max(1e-6, Math.min(grid.cellSizeX, grid.cellSizeY)),
  );
  for (let radius = 1; radius <= maxRadius; radius += 1) {
    let best = -1;
    let bestDistanceSq = Number.POSITIVE_INFINITY;
    for (let row = originRow - radius; row <= originRow + radius; row += 1) {
      for (let col = originCol - radius; col <= originCol + radius; col += 1) {
        if (
          row < 0 ||
          row >= grid.nrow ||
          col < 0 ||
          col >= grid.ncol ||
          (row !== originRow - radius &&
            row !== originRow + radius &&
            col !== originCol - radius &&
            col !== originCol + radius)
        ) {
          continue;
        }
        const index = row * grid.ncol + col;
        if (!isWalkable(grid, index, config, walkable, roughness)) {
          continue;
        }
        const distanceSq = (row - originRow) ** 2 + (col - originCol) ** 2;
        if (distanceSq < bestDistanceSq) {
          bestDistanceSq = distanceSq;
          best = index;
        }
      }
    }
    if (best >= 0) {
      return best;
    }
  }
  return -1;
}

function isWalkable(
  grid: HeightfieldPathGrid,
  index: number,
  config: PlannerConfig,
  walkable: Int8Array,
  roughness: Float32Array,
): boolean {
  const cached = walkable[index];
  if (cached >= 0) {
    return cached === 1;
  }

  const surfaceRoughness = localRoughness(grid, index, config, roughness);
  const ok = Number.isFinite(surfaceRoughness) && surfaceRoughness <= config.maxStepHeight;
  walkable[index] = ok ? 1 : 0;
  return ok;
}

function localRoughness(
  grid: HeightfieldPathGrid,
  index: number,
  config: PlannerConfig,
  roughness: Float32Array,
): number {
  const cached = roughness[index];
  if (Number.isFinite(cached)) {
    return cached;
  }

  const centerCol = index % grid.ncol;
  const centerRow = Math.floor(index / grid.ncol);
  const radius = Math.max(
    0,
    Math.floor(config.footprintRadius / Math.max(1e-6, Math.min(grid.cellSizeX, grid.cellSizeY))),
  );
  let minHeight = Number.POSITIVE_INFINITY;
  let maxHeight = Number.NEGATIVE_INFINITY;
  for (let row = centerRow - radius; row <= centerRow + radius; row += 1) {
    if (row < 0 || row >= grid.nrow) {
      roughness[index] = Number.POSITIVE_INFINITY;
      return roughness[index];
    }
    for (let col = centerCol - radius; col <= centerCol + radius; col += 1) {
      if (col < 0 || col >= grid.ncol) {
        roughness[index] = Number.POSITIVE_INFINITY;
        return roughness[index];
      }
      const height = heightAt(grid, row * grid.ncol + col);
      minHeight = Math.min(minHeight, height);
      maxHeight = Math.max(maxHeight, height);
    }
  }
  roughness[index] = maxHeight - minHeight;
  return roughness[index];
}

function edgeCost(
  grid: HeightfieldPathGrid,
  from: number,
  to: number,
  config: PlannerConfig,
  roughness: Float32Array,
): number {
  const fromCol = from % grid.ncol;
  const fromRow = Math.floor(from / grid.ncol);
  const toCol = to % grid.ncol;
  const toRow = Math.floor(to / grid.ncol);
  const dx = (toCol - fromCol) * grid.cellSizeX;
  const dy = (toRow - fromRow) * grid.cellSizeY;
  const horizontal = Math.hypot(dx, dy);
  const dz = Math.abs(heightAt(grid, to) - heightAt(grid, from));
  const slope = dz / Math.max(1e-6, horizontal);
  if (dz > config.maxStepHeight || slope > config.maxSlope) {
    return Number.POSITIVE_INFINITY;
  }
  const surfaceRoughness = localRoughness(grid, to, config, roughness);
  if (!Number.isFinite(surfaceRoughness)) {
    return Number.POSITIVE_INFINITY;
  }
  return horizontal * (
    1 +
    config.slopeCostWeight * slope * slope +
    config.roughnessCostWeight * surfaceRoughness
  );
}

function heuristic(grid: HeightfieldPathGrid, from: number, to: number): number {
  const fromCol = from % grid.ncol;
  const fromRow = Math.floor(from / grid.ncol);
  const toCol = to % grid.ncol;
  const toRow = Math.floor(to / grid.ncol);
  return Math.hypot((toCol - fromCol) * grid.cellSizeX, (toRow - fromRow) * grid.cellSizeY);
}

function reconstructPath(cameFrom: Int32Array, goal: number): number[] {
  const path = [goal];
  let current = goal;
  while (cameFrom[current] >= 0) {
    current = cameFrom[current];
    path.push(current);
  }
  path.reverse();
  return path;
}

function indicesToWaypoints(
  grid: HeightfieldPathGrid,
  indices: readonly number[],
  waypointSpacing: number,
): Vec3[] {
  if (indices.length === 0) {
    return [];
  }
  const points: Vec3[] = [];
  let last = indexToWorld(grid, indices[0]);
  points.push([last[0], last[1], last[2] + 0.08]);
  for (let i = 1; i < indices.length - 1; i += 1) {
    const point = indexToWorld(grid, indices[i]);
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) < waypointSpacing) {
      continue;
    }
    points.push([point[0], point[1], point[2] + 0.08]);
    last = point;
  }
  const end = indexToWorld(grid, indices[indices.length - 1]);
  points.push([end[0], end[1], end[2] + 0.08]);
  return points;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

class BinaryHeap {
  private readonly nodes: number[] = [];
  private readonly priorities: number[] = [];

  isEmpty(): boolean {
    return this.nodes.length === 0;
  }

  push(node: number, priority: number): void {
    this.nodes.push(node);
    this.priorities.push(priority);
    this.bubbleUp(this.nodes.length - 1);
  }

  pop(): number {
    if (this.nodes.length === 0) {
      return -1;
    }
    const node = this.nodes[0];
    const lastNode = this.nodes.pop()!;
    const lastPriority = this.priorities.pop()!;
    if (this.nodes.length > 0) {
      this.nodes[0] = lastNode;
      this.priorities[0] = lastPriority;
      this.sinkDown(0);
    }
    return node;
  }

  private bubbleUp(index: number): void {
    let child = index;
    while (child > 0) {
      const parent = Math.floor((child - 1) / 2);
      if (this.priorities[parent] <= this.priorities[child]) {
        return;
      }
      this.swap(parent, child);
      child = parent;
    }
  }

  private sinkDown(index: number): void {
    let parent = index;
    while (true) {
      const left = parent * 2 + 1;
      const right = left + 1;
      let smallest = parent;
      if (left < this.nodes.length && this.priorities[left] < this.priorities[smallest]) {
        smallest = left;
      }
      if (right < this.nodes.length && this.priorities[right] < this.priorities[smallest]) {
        smallest = right;
      }
      if (smallest === parent) {
        return;
      }
      this.swap(parent, smallest);
      parent = smallest;
    }
  }

  private swap(a: number, b: number): void {
    [this.nodes[a], this.nodes[b]] = [this.nodes[b], this.nodes[a]];
    [this.priorities[a], this.priorities[b]] = [this.priorities[b], this.priorities[a]];
  }
}
