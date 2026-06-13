import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import loadMujoco from "@mujoco/mujoco";

const packageDir = path.dirname(fileURLToPath(import.meta.resolve("@mujoco/mujoco")));
const wasmPath = path.join(packageDir, "mujoco.wasm");

const sceneXmlPath = process.argv[2] ?? "public/envs/g1_flat/scene.xml";
const assetDir = path.dirname(sceneXmlPath);
const rootDir = process.cwd();
const dumpScene = process.argv.includes("--dump-scene");

const enumValue = (value) => (typeof value === "object" && value !== null ? value.value : value);
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const formatVec = (v) => v.map((value) => value.toFixed(3)).join(",");
const stripRobot = (name) => name.replace(/^robot\//, "");

function meshFileNames(xml) {
  const names = [];
  const re = /<mesh\b[^>]*\bfile="([^"]+)"/g;
  for (let match = re.exec(xml); match; match = re.exec(xml)) {
    names.push(match[1]);
  }
  return names;
}

function expectedBodyForMesh(meshName) {
  const leaf = stripRobot(meshName);
  const aliases = {
    pelvis_contour_link: "pelvis",
    logo_link: "torso_link",
    head_link: "torso_link",
    left_rubber_hand: "left_wrist_yaw_link",
    right_rubber_hand: "right_wrist_yaw_link",
  };
  return `robot/${aliases[leaf] ?? leaf}`;
}

function transformPoint(rotation, position, point, mode) {
  const [x, y, z] = point;
  if (mode === "transpose") {
    return [
      rotation[0] * x + rotation[3] * y + rotation[6] * z + position[0],
      rotation[1] * x + rotation[4] * y + rotation[7] * z + position[1],
      rotation[2] * x + rotation[5] * y + rotation[8] * z + position[2],
    ];
  }
  return [
    rotation[0] * x + rotation[1] * y + rotation[2] * z + position[0],
    rotation[3] * x + rotation[4] * y + rotation[5] * z + position[1],
    rotation[6] * x + rotation[7] * y + rotation[8] * z + position[2],
  ];
}

function distanceToAabb(point, aabb) {
  let squared = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const value = point[axis];
    const delta =
      value < aabb.min[axis] ? aabb.min[axis] - value : value > aabb.max[axis] ? value - aabb.max[axis] : 0;
    squared += delta * delta;
  }
  return Math.sqrt(squared);
}

function meshFaceStats(model, meshId) {
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < faceNum * 3; i += 1) {
    const value = model.mesh_face[faceAdr * 3 + i];
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { min, max };
}

function meshWorldAabb(model, data, geomId, faceMode, matrixMode) {
  const meshId = model.geom_dataid[geomId];
  const vertAdr = model.mesh_vertadr[meshId];
  const vertNum = model.mesh_vertnum[meshId];
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];
  const position = [
    data.geom_xpos[geomId * 3],
    data.geom_xpos[geomId * 3 + 1],
    data.geom_xpos[geomId * 3 + 2],
  ];
  const rotation = Array.from(data.geom_xmat.slice(geomId * 9, geomId * 9 + 9));
  const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  for (let face = 0; face < faceNum; face += 1) {
    for (let corner = 0; corner < 3; corner += 1) {
      const faceIndex = model.mesh_face[(faceAdr + face) * 3 + corner];
      const vertexIndex = faceMode === "global" ? faceIndex : vertAdr + faceIndex;
      if (vertexIndex < vertAdr || vertexIndex >= vertAdr + vertNum) {
        return null;
      }
      const source = vertexIndex * 3;
      const world = transformPoint(
        rotation,
        position,
        [
          model.mesh_vert[source],
          model.mesh_vert[source + 1],
          model.mesh_vert[source + 2],
        ],
        matrixMode,
      );
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], world[axis]);
        max[axis] = Math.max(max[axis], world[axis]);
      }
    }
  }

  return {
    min,
    max,
    center: min.map((value, axis) => 0.5 * (value + max[axis])),
    size: min.map((value, axis) => max[axis] - value),
  };
}

function getName(mujoco, model, objType, id) {
  return mujoco.mj_id2name(model, objType, id) || `<unnamed:${id}>`;
}

function scoreMode({
  mujoco,
  model,
  data,
  bodyPositions,
  robotBodies,
  childrenByBody,
  matrixMode,
  faceMode,
}) {
  const OBJ_GEOM = enumValue(mujoco.mjtObj.mjOBJ_GEOM);
  const OBJ_MESH = enumValue(mujoco.mjtObj.mjOBJ_MESH);
  const OBJ_BODY = enumValue(mujoco.mjtObj.mjOBJ_BODY);
  const GEOM_MESH = enumValue(mujoco.mjtGeom.mjGEOM_MESH);

  const rows = [];
  for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
    if (model.geom_type[geomId] !== GEOM_MESH || model.geom_group[geomId] !== 2) {
      continue;
    }
    const meshId = model.geom_dataid[geomId];
    const aabb = meshWorldAabb(model, data, geomId, faceMode, matrixMode);
    const meshName = getName(mujoco, model, OBJ_MESH, meshId);
    const bodyId = model.geom_bodyid[geomId];
    const bodyName = getName(mujoco, model, OBJ_BODY, bodyId);
    const expectedBody = expectedBodyForMesh(meshName);
    const expectedPosition = bodyPositions.get(expectedBody);
    const expectedChildren = childrenByBody.get(expectedBody) ?? [];
    const geomPosition = [
      data.geom_xpos[geomId * 3],
      data.geom_xpos[geomId * 3 + 1],
      data.geom_xpos[geomId * 3 + 2],
    ];

    if (!aabb) {
      rows.push({
        status: "bad-face-index",
        geomId,
        meshName,
        bodyName,
        expectedBody,
        distanceToExpected: Number.POSITIVE_INFINITY,
        distanceToGeom: Number.POSITIVE_INFINITY,
      });
      continue;
    }

    let nearestBody = "";
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const robotBody of robotBodies) {
      const d = distance(aabb.center, robotBody.position);
      if (d < nearestDistance) {
        nearestDistance = d;
        nearestBody = robotBody.name;
      }
    }

    rows.push({
      status: "ok",
      geomId,
      geomName: getName(mujoco, model, OBJ_GEOM, geomId),
      meshName,
      bodyName,
      expectedBody,
      nearestBody,
      nearestDistance,
      distanceToExpected: expectedPosition ? distance(aabb.center, expectedPosition) : Number.NaN,
      distanceToGeom: distance(aabb.center, geomPosition),
      childAabbDistance:
        expectedChildren.length > 0
          ? Math.min(...expectedChildren.map((child) => distanceToAabb(child.position, aabb)))
          : Number.NaN,
      center: aabb.center,
      size: aabb.size,
    });
  }

  const finite = rows.filter((row) => Number.isFinite(row.distanceToExpected));
  const childFinite = rows.filter((row) => Number.isFinite(row.childAabbDistance));
  const sorted = [...finite].sort((a, b) => b.distanceToExpected - a.distanceToExpected);
  const childSorted = [...childFinite].sort((a, b) => b.childAabbDistance - a.childAabbDistance);
  const sum = finite.reduce((acc, row) => acc + row.distanceToExpected, 0);
  const childSum = childFinite.reduce((acc, row) => acc + row.childAabbDistance, 0);
  const mismatchCount = finite.filter((row) => row.nearestBody !== row.expectedBody).length;
  return {
    faceMode,
    matrixMode,
    count: rows.length,
    invalidFaceCount: rows.length - finite.length,
    avgDistanceToExpected: sum / Math.max(1, finite.length),
    maxDistanceToExpected: sorted[0]?.distanceToExpected ?? 0,
    nearestMismatchCount: mismatchCount,
    avgChildAabbDistance: childSum / Math.max(1, childFinite.length),
    maxChildAabbDistance: childSorted[0]?.childAabbDistance ?? 0,
    rows,
    worst: sorted.slice(0, 10),
    worstChild: childSorted.slice(0, 10),
  };
}

async function main() {
  const xml = await readFile(sceneXmlPath, "utf8");
  const mujoco = await loadMujoco({
    locateFile: (file) => (file.endsWith(".wasm") ? wasmPath : file),
  });
  const vfs = new mujoco.MjVFS();
  for (const fileName of meshFileNames(xml)) {
    const bytes = await readFile(path.join(rootDir, assetDir, "assets", fileName));
    vfs.addBuffer(`assets/${fileName}`, new Uint8Array(bytes));
    vfs.addBuffer(fileName, new Uint8Array(bytes));
  }

  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);

  const OBJ_MESH = enumValue(mujoco.mjtObj.mjOBJ_MESH);
  const OBJ_GEOM = enumValue(mujoco.mjtObj.mjOBJ_GEOM);
  const OBJ_BODY = enumValue(mujoco.mjtObj.mjOBJ_BODY);
  const GEOM_MESH = enumValue(mujoco.mjtGeom.mjGEOM_MESH);

  if (dumpScene) {
    const option = new mujoco.MjvOption();
    const perturb = new mujoco.MjvPerturb();
    const camera = new mujoco.MjvCamera();
    const scene = new mujoco.MjvScene(model, 2 ** 15);
    for (let i = 0; i < option.geomgroup.length; i += 1) {
      option.geomgroup[i] = i <= 2 ? 1 : 0;
    }
    mujoco.mjv_updateScene(
      model,
      data,
      option,
      perturb,
      camera,
      enumValue(mujoco.mjtCatBit.mjCAT_ALL),
      scene,
    );

    console.log("Scene geoms");
    const geoms = scene.geoms;
    for (let sceneIndex = 0; sceneIndex < geoms.size(); sceneIndex += 1) {
      const geom = geoms.get(sceneIndex);
      if (geom.type !== GEOM_MESH) {
        geom.delete();
        continue;
      }
      const modelGeomId = geom.objtype === OBJ_GEOM ? geom.objid : -1;
      const modelMeshId =
        geom.objtype === OBJ_GEOM && geom.objid >= 0 ? model.geom_dataid[geom.objid] : geom.dataid;
      const modelBodyId = modelGeomId >= 0 ? model.geom_bodyid[modelGeomId] : -1;
      console.log(
        [
          `scene=${String(sceneIndex).padStart(2, "0")}`,
          `objtype=${geom.objtype}`,
          `objid=${String(geom.objid).padStart(2, "0")}`,
          `dataid=${String(geom.dataid).padStart(2, "0")}`,
          `geom=${modelGeomId >= 0 ? getName(mujoco, model, OBJ_GEOM, modelGeomId) : "n/a"}`,
          `mesh=${modelMeshId >= 0 ? getName(mujoco, model, OBJ_MESH, modelMeshId) : "n/a"}`,
          `body=${modelBodyId >= 0 ? getName(mujoco, model, OBJ_BODY, modelBodyId) : "n/a"}`,
        ].join("  "),
      );
      geom.delete();
    }
    geoms.delete();
    scene.delete();
    camera.delete();
    perturb.delete();
    option.delete();
  }

  const bodyPositions = new Map();
  const robotBodies = [];
  const childrenByBody = new Map();
  for (let bodyId = 1; bodyId < model.nbody; bodyId += 1) {
    const name = getName(mujoco, model, OBJ_BODY, bodyId);
    const position = [
      data.xpos[bodyId * 3],
      data.xpos[bodyId * 3 + 1],
      data.xpos[bodyId * 3 + 2],
    ];
    bodyPositions.set(name, position);
    if (name.startsWith("robot/")) {
      robotBodies.push({ id: bodyId, name, position });
      const parentId = model.body_parentid[bodyId];
      const parentName = getName(mujoco, model, OBJ_BODY, parentId);
      if (parentName.startsWith("robot/")) {
        const children = childrenByBody.get(parentName) ?? [];
        children.push({ id: bodyId, name, position });
        childrenByBody.set(parentName, children);
      }
    }
  }

  console.log("Face index range check");
  for (let meshId = 0; meshId < model.nmesh; meshId += 1) {
    const stats = meshFaceStats(model, meshId);
    const meshName = getName(mujoco, model, OBJ_MESH, meshId);
    const vertAdr = model.mesh_vertadr[meshId];
    const vertNum = model.mesh_vertnum[meshId];
    const local = stats.min >= 0 && stats.max < vertNum;
    const global = stats.min >= vertAdr && stats.max < vertAdr + vertNum;
    console.log(
      [
        String(meshId).padStart(2, "0"),
        meshName.padEnd(31, " "),
        `verts=${String(vertNum).padStart(5, " ")}`,
        `face=[${stats.min},${stats.max}]`,
        `local=${local ? "yes" : "no "}`,
        `global=${global ? "yes" : "no "}`,
      ].join("  "),
    );
  }

  const scores = [];
  for (const faceMode of ["local", "global"]) {
    for (const matrixMode of ["row", "transpose"]) {
      scores.push(
        scoreMode({
          mujoco,
          model,
          data,
          bodyPositions,
          robotBodies,
          childrenByBody,
          faceMode,
          matrixMode,
        }),
      );
    }
  }

  console.log("\nRender mapping score");
  for (const score of scores) {
    console.log(
      [
        `face=${score.faceMode.padEnd(6, " ")}`,
        `matrix=${score.matrixMode.padEnd(9, " ")}`,
        `invalid=${String(score.invalidFaceCount).padStart(2, " ")}`,
        `nearestMismatch=${String(score.nearestMismatchCount).padStart(2, " ")}/${score.count}`,
        `avgExpected=${score.avgDistanceToExpected.toFixed(3)}m`,
        `maxExpected=${score.maxDistanceToExpected.toFixed(3)}m`,
        `avgChildAabb=${score.avgChildAabbDistance.toFixed(3)}m`,
        `maxChildAabb=${score.maxChildAabbDistance.toFixed(3)}m`,
      ].join("  "),
    );
  }

  const best = scores
    .filter((score) => score.invalidFaceCount === 0)
    .sort((a, b) => {
      if (a.nearestMismatchCount !== b.nearestMismatchCount) {
        return a.nearestMismatchCount - b.nearestMismatchCount;
      }
      if (a.avgChildAabbDistance !== b.avgChildAabbDistance) {
        return a.avgChildAabbDistance - b.avgChildAabbDistance;
      }
      return a.avgDistanceToExpected - b.avgDistanceToExpected;
    })[0];

  console.log(`\nBest mode: face=${best.faceMode}, matrix=${best.matrixMode}`);
  console.log("Worst parts by expected-body distance");
  for (const row of best.worst) {
    console.log(
      [
        row.meshName.padEnd(31, " "),
        `body=${stripRobot(row.bodyName).padEnd(23, " ")}`,
        `expected=${stripRobot(row.expectedBody).padEnd(19, " ")}`,
        `nearest=${stripRobot(row.nearestBody).padEnd(23, " ")}`,
        `dExpected=${row.distanceToExpected.toFixed(3)}m`,
        `dGeom=${row.distanceToGeom.toFixed(3)}m`,
        `dChildAabb=${Number.isFinite(row.childAabbDistance) ? `${row.childAabbDistance.toFixed(3)}m` : "n/a"}`,
        `center=[${formatVec(row.center)}]`,
        `size=[${formatVec(row.size)}]`,
      ].join("  "),
    );
  }

  console.log("\nWorst link child-to-AABB distances");
  for (const row of best.worstChild) {
    console.log(
      [
        row.meshName.padEnd(31, " "),
        `expected=${stripRobot(row.expectedBody).padEnd(23, " ")}`,
        `dChildAabb=${row.childAabbDistance.toFixed(3)}m`,
        `center=[${formatVec(row.center)}]`,
        `size=[${formatVec(row.size)}]`,
      ].join("  "),
    );
  }

  console.log("\n--- Head Link Symmetry and Alignment Analysis ---");
  for (const matrixMode of ["row", "transpose"]) {
    let headGeomId = -1;
    const OBJ_GEOM = enumValue(mujoco.mjtObj.mjOBJ_GEOM);
    const OBJ_MESH = enumValue(mujoco.mjtObj.mjOBJ_MESH);
    const GEOM_MESH = enumValue(mujoco.mjtGeom.mjGEOM_MESH);
    for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
      if (model.geom_type[geomId] === GEOM_MESH) {
        const meshId = model.geom_dataid[geomId];
        const meshName = getName(mujoco, model, OBJ_MESH, meshId);
        if (meshName === "robot/head_link") {
          headGeomId = geomId;
          break;
        }
      }
    }

    if (headGeomId !== -1) {
      const aabb = meshWorldAabb(model, data, headGeomId, "local", matrixMode);
      if (aabb) {
        console.log(`[Matrix Mode: ${matrixMode}]`);
        console.log(`  AABB Center: X=${aabb.center[0].toFixed(6)}, Y=${aabb.center[1].toFixed(6)} (Symmetry Check), Z=${aabb.center[2].toFixed(6)}`);
        console.log(`  AABB Size:   X=${aabb.size[0].toFixed(6)}, Y=${aabb.size[1].toFixed(6)}, Z=${aabb.size[2].toFixed(6)}`);
        console.log(`  Y Range:     [${aabb.min[1].toFixed(6)}, ${aabb.max[1].toFixed(6)}]`);
        
        const symmetryDeviation = Math.abs(aabb.min[1] + aabb.max[1]);
        console.log(`  Symmetry Deviation (|Y_min + Y_max|): ${symmetryDeviation.toFixed(6)}m`);
        if (symmetryDeviation < 0.001) {
          console.log("  => Result: EXCELLENT SYMMETRY (Face points forward perfectly!)");
        } else if (symmetryDeviation < 0.005) {
          console.log("  => Result: GOOD SYMMETRY (Face is highly symmetric and likely forward)");
        } else {
          console.log("  => Result: ASYMMETRIC (Mesh is rotated, tilted, or offset!)");
        }
      } else {
        console.log(`[Matrix Mode: ${matrixMode}] Failed to compute AABB`);
      }
    } else {
      console.log("Could not find geom for robot/head_link");
      break;
    }
  }

  data.delete();
  model.delete();
  vfs.delete();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
