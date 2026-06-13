import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { EnvDefinition } from "../envs/types";
import type { GaussianSplatVisualSource } from "./gaussianHeightfield";
import { publicUrl } from "../publicUrl";

type MujocoModule = any;
type MujocoModel = any;
type MujocoData = any;

const DEFAULT_CAMERA_NEAR = 0.01;
const DEFAULT_CAMERA_FAR = 120;
const GAUSSIAN_CAMERA_NEAR = 0.05;
const PICKER_MARKER_RADIUS = 0.085;
const ROUTE_GOAL_MARKER_RADIUS = 0.1;
const ROUTE_LINE_RADIUS = 0.045;

export interface RenderPart {
  key: string;
  label: string;
  meshName: string;
  bodyName: string;
  visible: boolean;
}

export interface SpawnPickerMoveInput {
  forward: number;
  right: number;
  up: number;
  speed: number;
}

export interface SpawnPickerVisualOptions {
  color?: number;
  emissive?: number;
}

interface VisualMeshManifest {
  version: number;
  parts: VisualMeshPart[];
}

interface VisualMeshPart {
  key?: string;
  label?: string;
  bodyName: string;
  meshName: string;
  url: string;
  pos?: number[];
  quat?: number[];
  rgba?: number[];
}

interface VisualMeshInstance {
  key: string;
  bodyId: number;
  mesh: THREE.Mesh;
  localMatrix: THREE.Matrix4;
}

export class MujocoRenderer {
  readonly renderer: THREE.WebGLRenderer;

  private readonly mujoco: MujocoModule;
  private readonly host: HTMLElement;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly dynamicGroup = new THREE.Group();
  private readonly visualMeshGroup = new THREE.Group();
  private readonly gaussianSplatGroup = new THREE.Group();
  private readonly skeletonGroup = new THREE.Group();
  private readonly routeLineGroup = new THREE.Group();
  private readonly meshes = new Map<string, THREE.Mesh>();
  private readonly visualMeshes = new Map<string, THREE.Mesh>();
  private readonly geometryCache = new Map<string, THREE.BufferGeometry>();
  private readonly visualGeometryCache = new Map<string, THREE.BufferGeometry>();
  private readonly materialCache = new Map<string, THREE.Material>();
  private readonly resizeObserver: ResizeObserver;
  private readonly bodyWorldMatrix = new THREE.Matrix4();

  private model: MujocoModel | null = null;
  private data: MujocoData | null = null;
  private mjvOption: any;
  private mjvPerturb: any;
  private mjvCamera: any;
  private mjvScene: any;
  private env: EnvDefinition | null = null;
  private bodyPairs: Array<[number, number]> = [];
  private skeletonBodyIds: number[] = [];
  private skeletonLines: THREE.LineSegments | null = null;
  private skeletonPoints: THREE.Points | null = null;
  private visualMeshInstances: VisualMeshInstance[] = [];
  private readonly renderParts = new Map<string, RenderPart>();
  private readonly partVisibility = new Map<string, boolean>();
  private meshesEnabled = true;
  private heightfieldEnabled = true;
  private routeLine: THREE.Mesh | null = null;
  private sparkRenderer: any | null = null;
  private gaussianSplatMesh: any | null = null;
  private spawnPickerEnabled = false;
  private spawnPickerMarker: THREE.Mesh | null = null;
  private readonly spawnPickerPosition = new THREE.Vector3();
  private routeGoalMarker: THREE.Mesh | null = null;

  constructor(mujoco: MujocoModule, host: HTMLElement) {
    this.mujoco = mujoco;
    this.host = host;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111314);
    this.scene.add(this.dynamicGroup);
    this.scene.add(this.visualMeshGroup);
    this.scene.add(this.gaussianSplatGroup);
    this.scene.add(this.skeletonGroup);
    this.scene.add(this.routeLineGroup);
    this.skeletonGroup.visible = false;
    this.gaussianSplatGroup.matrixAutoUpdate = false;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.host.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(42, 1, DEFAULT_CAMERA_NEAR, DEFAULT_CAMERA_FAR);
    this.camera.up.set(0, 0, 1);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    let startX = 0;
    let startY = 0;
    let startTime = 0;

    this.renderer.domElement.addEventListener("pointerdown", (event) => {
      startX = event.clientX;
      startY = event.clientY;
      startTime = Date.now();
    });

    this.renderer.domElement.addEventListener("pointerup", (event) => {
      if (!this.spawnPickerEnabled) {
        return;
      }
      const dx = event.clientX - startX;
      const dy = event.clientY - startY;
      const dist = Math.hypot(dx, dy);
      const duration = Date.now() - startTime;

      // Only count as a click if pointer moved less than 4 pixels and was pressed for less than 300ms
      if (dist < 4 && duration < 300) {
        this.handleCanvasClick(event);
      }
    });

    this.addLights();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.host);
    this.resize();
  }

  async load(
    env: EnvDefinition,
    model: MujocoModel,
    data: MujocoData,
    gaussianSplatVisualSource: GaussianSplatVisualSource | null = null,
  ): Promise<void> {
    this.disposeMujocoScene();
    this.clearDynamicMeshes();
    this.env = env;
    this.model = model;
    this.data = data;
    this.mjvPerturb = new this.mujoco.MjvPerturb();
    this.mjvOption = new this.mujoco.MjvOption();
    this.configureDefaultGeomGroups();
    this.mjvCamera = new this.mujoco.MjvCamera();
    this.mjvScene = new this.mujoco.MjvScene(model, 2 ** 15);
    this.buildRenderParts();
    await this.loadVisualMeshes(env);
    await this.loadGaussianSplatVisual(gaussianSplatVisualSource);
    this.configureCameraClipRange(gaussianSplatVisualSource);
    this.buildSkeleton();
    this.resetCamera();
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    this.disposeMujocoScene();
    this.clearDynamicMeshes();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  setContactVisualization(enabled: boolean): void {
    if (!this.mjvOption) {
      return;
    }
    this.mjvOption.geomgroup[3] = enabled ? 1 : 0;
    const contactPoint = this.mujoco.mjtVisFlag.mjVIS_CONTACTPOINT?.value;
    const contactForce = this.mujoco.mjtVisFlag.mjVIS_CONTACTFORCE?.value;
    if (contactPoint !== undefined) {
      this.mjvOption.flags[contactPoint] = enabled;
    }
    if (contactForce !== undefined) {
      this.mjvOption.flags[contactForce] = enabled;
    }
  }

  setMeshVisualization(enabled: boolean): void {
    this.meshesEnabled = enabled;
    for (const [meshKey, mesh] of this.meshes) {
      mesh.visible = this.visibleForMeshKey(meshKey);
    }
    for (const [meshKey, mesh] of this.visualMeshes) {
      mesh.visible = this.visibleForMeshKey(meshKey);
    }
    this.gaussianSplatGroup.visible = enabled;
    if (this.gaussianSplatMesh) {
      this.gaussianSplatMesh.visible = enabled;
    }
  }

  setHeightfieldVisualization(enabled: boolean): void {
    this.heightfieldEnabled = enabled;
    for (const [meshKey, mesh] of this.meshes) {
      mesh.visible = this.visibleForMeshKey(meshKey);
    }
  }

  setSkeletonVisualization(enabled: boolean): void {
    this.skeletonGroup.visible = enabled;
  }

  getRenderParts(): RenderPart[] {
    return Array.from(this.renderParts.values()).map((part) => ({
      ...part,
      visible: this.partVisibility.get(part.key) ?? true,
    }));
  }

  setPartVisibility(key: string, visible: boolean): void {
    const part = this.renderParts.get(key);
    if (!part) {
      return;
    }
    part.visible = visible;
    this.partVisibility.set(key, visible);
    const mesh = this.meshes.get(key);
    if (mesh) {
      mesh.visible = this.visibleForMeshKey(key);
    }
    const visualMesh = this.visualMeshes.get(key);
    if (visualMesh) {
      visualMesh.visible = this.visibleForMeshKey(key);
    }
  }

  setAllPartVisibility(visible: boolean): void {
    for (const part of this.renderParts.values()) {
      part.visible = visible;
      this.partVisibility.set(part.key, visible);
    }
    for (const [meshKey, mesh] of this.meshes) {
      mesh.visible = this.visibleForMeshKey(meshKey);
    }
    for (const [meshKey, mesh] of this.visualMeshes) {
      mesh.visible = this.visibleForMeshKey(meshKey);
    }
  }

  refreshMujocoGeometry(): void {
    for (const geometry of this.geometryCache.values()) {
      geometry.dispose();
    }
    this.geometryCache.clear();
  }

  setRouteLine(points: ReadonlyArray<readonly [number, number, number]>): void {
    this.clearRouteLine();
    if (points.length < 2) {
      return;
    }
    const curve = new THREE.CatmullRomCurve3(
      points.map((point) => new THREE.Vector3(point[0], point[1], point[2] + ROUTE_LINE_RADIUS)),
      false,
      "centripetal",
    );
    const geometry = new THREE.TubeGeometry(
      curve,
      Math.max(16, (points.length - 1) * 12),
      ROUTE_LINE_RADIUS,
      10,
      false,
    );
    const material = new THREE.MeshBasicMaterial({
      color: 0xffd166,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    });
    this.routeLine = new THREE.Mesh(geometry, material);
    this.routeLine.frustumCulled = false;
    this.routeLine.renderOrder = 9;
    this.routeLineGroup.add(this.routeLine);
  }

  setSpawnPicker(
    enabled: boolean,
    position?: readonly [number, number, number],
    visualOptions: SpawnPickerVisualOptions = {},
  ): void {
    this.spawnPickerEnabled = enabled;
    if (position) {
      this.spawnPickerPosition.set(position[0], position[1], position[2]);
    }
    if (!enabled && !this.spawnPickerMarker) {
      return;
    }
    const marker = this.ensureSpawnPickerMarker();
    marker.visible = enabled;
    marker.position.copy(this.spawnPickerPosition);
    this.applyMarkerMaterial(
      marker,
      visualOptions.color ?? 0xffd166,
      visualOptions.emissive ?? 0x6a3f00,
    );
  }

  setSpawnPickerPosition(position: readonly [number, number, number]): void {
    this.spawnPickerPosition.set(position[0], position[1], position[2]);
    if (this.spawnPickerMarker) {
      this.spawnPickerMarker.position.copy(this.spawnPickerPosition);
    }
  }

  getSpawnPickerPosition(): [number, number, number] {
    return [
      this.spawnPickerPosition.x,
      this.spawnPickerPosition.y,
      this.spawnPickerPosition.z,
    ];
  }

  setRouteGoalMarker(
    position: readonly [number, number, number] | null,
    visualOptions: SpawnPickerVisualOptions = {},
  ): void {
    if (!position) {
      this.clearRouteGoalMarker();
      return;
    }
    const marker = this.ensureRouteGoalMarker();
    marker.position.set(position[0], position[1], position[2]);
    marker.visible = true;
    this.applyMarkerMaterial(
      marker,
      visualOptions.color ?? 0x5bd6c6,
      visualOptions.emissive ?? 0x0c5a52,
    );
  }

  moveSpawnPicker(input: SpawnPickerMoveInput, dt: number): [number, number, number] {
    if (!this.spawnPickerEnabled) {
      return this.getSpawnPickerPosition();
    }
    const clampedDt = Math.min(0.05, Math.max(0, dt));
    const step = input.speed * clampedDt;
    if (step <= 0) {
      return this.getSpawnPickerPosition();
    }

    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    forward.z = 0;
    if (forward.lengthSq() < 1e-8) {
      forward.set(1, 0, 0);
    } else {
      forward.normalize();
    }

    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    right.z = 0;
    if (right.lengthSq() < 1e-8) {
      right.crossVectors(forward, this.camera.up);
    }
    right.normalize();

    this.spawnPickerPosition.addScaledVector(forward, input.forward * step);
    this.spawnPickerPosition.addScaledVector(right, input.right * step);
    this.spawnPickerPosition.z = Math.max(0.05, this.spawnPickerPosition.z + input.up * step);
    this.spawnPickerMarker?.position.copy(this.spawnPickerPosition);
    return this.getSpawnPickerPosition();
  }

  update(): void {
    if (!this.model || !this.data || !this.mjvScene) {
      return;
    }

    this.followBody();
    this.controls.update();
    this.mujoco.mjv_updateScene(
      this.model,
      this.data,
      this.mjvOption,
      this.mjvPerturb,
      this.mjvCamera,
      this.mujoco.mjtCatBit.mjCAT_ALL.value,
      this.mjvScene,
    );

    const geoms = this.mjvScene.geoms;
    const geomCount = geoms.size();
    const activeMeshKeys = new Set<string>();
    for (let i = 0; i < geomCount; i += 1) {
      const mjvGeom = geoms.get(i);
      const meshKey = this.meshKeyForGeom(i, mjvGeom);
      activeMeshKeys.add(meshKey);
      const mesh = this.meshForGeom(meshKey, mjvGeom);
      mesh.visible = this.visibleForMeshKey(meshKey, mjvGeom.type);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.set(
        mjvGeom.mat[0],
        mjvGeom.mat[1],
        mjvGeom.mat[2],
        mjvGeom.pos[0],
        mjvGeom.mat[3],
        mjvGeom.mat[4],
        mjvGeom.mat[5],
        mjvGeom.pos[1],
        mjvGeom.mat[6],
        mjvGeom.mat[7],
        mjvGeom.mat[8],
        mjvGeom.pos[2],
        0,
        0,
        0,
        1,
      );
      mesh.matrixWorldNeedsUpdate = true;
      if (
        this.gaussianSplatMesh &&
        mjvGeom.type === enumValue(this.mujoco.mjtGeom.mjGEOM_HFIELD)
      ) {
        this.gaussianSplatGroup.matrix.copy(mesh.matrix);
        this.gaussianSplatGroup.matrixWorldNeedsUpdate = true;
        this.gaussianSplatGroup.updateMatrixWorld(true);
        this.gaussianSplatMesh.updateMatrixWorld(true);
      }
      mjvGeom.delete();
    }
    geoms.delete();

    for (const [meshKey, mesh] of this.meshes) {
      mesh.visible = this.visibleForMeshKey(meshKey) && activeMeshKeys.has(meshKey);
    }
    this.updateVisualMeshes();
    this.updateSkeleton();
  }

  render(): void {
    if (this.sparkRenderer) {
      this.scene.updateMatrixWorld(true);
      this.sparkRenderer.render(this.scene, this.camera);
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  private addLights(): void {
    const ambient = new THREE.HemisphereLight(0xf5f0e8, 0x252a2c, 1.15);
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(-3.5, -4.5, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 18;
    key.shadow.camera.left = -5;
    key.shadow.camera.right = 5;
    key.shadow.camera.top = 5;
    key.shadow.camera.bottom = -5;
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x8bd7d2, 0.8);
    rim.position.set(4, 3, 3);
    this.scene.add(rim);
  }

  private configureDefaultGeomGroups(): void {
    const geomGroups = this.mjvOption.geomgroup;
    for (let i = 0; i < geomGroups.length; i += 1) {
      geomGroups[i] = i <= 2 ? 1 : 0;
    }
  }

  private resetCamera(): void {
    if (!this.env) {
      return;
    }
    const azimuth = (this.env.viewer.azimuthDeg * Math.PI) / 180;
    const elevation = (this.env.viewer.elevationDeg * Math.PI) / 180;
    const r = this.env.viewer.distance;
    const target = this.getFollowTarget();
    this.controls.target.set(target.x, target.y, target.z);
    this.camera.position.set(
      target.x + r * Math.cos(elevation) * Math.cos(azimuth),
      target.y + r * Math.cos(elevation) * Math.sin(azimuth),
      target.z + r * Math.sin(elevation),
    );
    this.controls.update();
  }

  private configureCameraClipRange(visualSource: GaussianSplatVisualSource | null): void {
    if (visualSource) {
      const radius = Math.max(1, visualSource.boundsRadius);
      this.camera.near = GAUSSIAN_CAMERA_NEAR;
      this.camera.far = Math.max(DEFAULT_CAMERA_FAR, roundUp(radius * 10 + 100, 100));
      this.controls.maxDistance = Math.max(50, this.camera.far * 0.45);
    } else {
      this.camera.near = DEFAULT_CAMERA_NEAR;
      this.camera.far = DEFAULT_CAMERA_FAR;
      this.controls.maxDistance = Number.POSITIVE_INFINITY;
    }
    this.camera.updateProjectionMatrix();
  }

  private followBody(): void {
    const target = this.getFollowTarget();
    const previousTarget = this.controls.target.clone();
    this.controls.target.lerp(target, 0.18);
    this.camera.position.add(this.controls.target.clone().sub(previousTarget));
  }

  private getFollowTarget(): THREE.Vector3 {
    if (this.spawnPickerEnabled) {
      return this.spawnPickerPosition.clone();
    }
    if (!this.env || !this.data) {
      return new THREE.Vector3(0, 0, 0.35);
    }
    const body = this.data.body(this.env.viewer.followBody);
    return new THREE.Vector3(body.xpos[0], body.xpos[1], body.xpos[2]);
  }

  private ensureSpawnPickerMarker(): THREE.Mesh {
    if (this.spawnPickerMarker) {
      return this.spawnPickerMarker;
    }
    const geometry = new THREE.SphereGeometry(PICKER_MARKER_RADIUS, 32, 18);
    const material = this.markerMaterial(0xffd166, 0x6a3f00);
    const marker = new THREE.Mesh(geometry, material);
    marker.castShadow = false;
    marker.receiveShadow = false;
    marker.visible = false;
    marker.renderOrder = 24;
    this.spawnPickerMarker = marker;
    this.scene.add(marker);
    return marker;
  }

  private ensureRouteGoalMarker(): THREE.Mesh {
    if (this.routeGoalMarker) {
      return this.routeGoalMarker;
    }
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(ROUTE_GOAL_MARKER_RADIUS, 32, 18),
      this.markerMaterial(0x5bd6c6, 0x0c5a52),
    );
    marker.castShadow = false;
    marker.receiveShadow = false;
    marker.visible = false;
    marker.renderOrder = 23;
    this.routeGoalMarker = marker;
    this.scene.add(marker);
    return marker;
  }

  private handleCanvasClick(event: PointerEvent): void {
    if (!this.spawnPickerEnabled || !this.model || !this.mujoco) {
      return;
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, this.camera);

    const intersects = raycaster.intersectObjects(this.dynamicGroup.children, true);
    for (const hit of intersects) {
      const mesh = hit.object as THREE.Mesh;
      const objid = mesh.userData?.objid;
      if (objid !== undefined && this.model) {
        const bodyId = this.model.geom_bodyid[objid];
        if (bodyId !== undefined && bodyId >= 0) {
          const bodyName = this.objectName(this.mujoco.mjtObj.mjOBJ_BODY, bodyId);
          if (bodyName && (bodyName.startsWith("robot") || bodyName.includes("robot"))) {
            continue; // Skip the robot itself
          }
        }
      }

      const newPos: [number, number, number] = [
        hit.point.x,
        hit.point.y,
        hit.point.z + PICKER_MARKER_RADIUS,
      ];
      this.setSpawnPickerPosition(newPos);
      this.host.dispatchEvent(new CustomEvent("mujoco-picker-click", {
        detail: { position: newPos },
        bubbles: true,
      }));
      break;
    }
  }

  private markerMaterial(color: number, emissive: number): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color,
      emissive,
      emissiveIntensity: 0.5,
      roughness: 0.35,
      metalness: 0.05,
      transparent: true,
      opacity: 0.92,
      depthTest: true,
    });
  }

  private applyMarkerMaterial(marker: THREE.Mesh, color: number, emissive: number): void {
    const material = marker.material as THREE.MeshStandardMaterial;
    material.color.setHex(color);
    material.emissive.setHex(emissive);
  }

  private buildRenderParts(): void {
    this.renderParts.clear();
    this.partVisibility.clear();
    if (!this.model) {
      return;
    }
    const meshType = enumValue(this.mujoco.mjtGeom.mjGEOM_MESH);
    for (let geomId = 0; geomId < this.model.ngeom; geomId += 1) {
      if (this.model.geom_type[geomId] !== meshType || this.model.geom_group[geomId] > 2) {
        continue;
      }
      const meshId = this.model.geom_dataid[geomId];
      if (meshId < 0) {
        continue;
      }
      const bodyId = this.model.geom_bodyid[geomId];
      const meshName = this.objectName(this.mujoco.mjtObj.mjOBJ_MESH, meshId);
      const bodyName = this.objectName(this.mujoco.mjtObj.mjOBJ_BODY, bodyId);
      const geomName = this.objectName(this.mujoco.mjtObj.mjOBJ_GEOM, geomId);
      const key = this.meshKeyForModelGeom(geomId);
      const part: RenderPart = {
        key,
        label: this.partLabel(meshName, bodyName, geomName),
        meshName,
        bodyName,
        visible: true,
      };
      this.renderParts.set(key, part);
      this.partVisibility.set(key, true);
    }
  }

  private async loadVisualMeshes(env: EnvDefinition): Promise<void> {
    const manifestUrl = env.render?.visualMeshManifestUrl;
    if (!manifestUrl || !this.model) {
      return;
    }

    const response = await fetch(manifestUrl);
    if (!response.ok) {
      throw new Error(`Failed to load visual mesh manifest: ${response.status} ${response.statusText}`);
    }
    const manifest = (await response.json()) as VisualMeshManifest;
    if (manifest.version !== 1 || !Array.isArray(manifest.parts)) {
      throw new Error("Unsupported visual mesh manifest");
    }

    const loader = new GLTFLoader();
    const uniqueUrls = [...new Set(manifest.parts.map((part) => publicUrl(part.url)))];
    await Promise.all(
      uniqueUrls.map(async (url) => {
        const gltf = await loader.loadAsync(url);
        const geometry = this.firstMeshGeometry(gltf.scene);
        geometry.computeBoundingSphere();
        this.visualGeometryCache.set(url, geometry);
      }),
    );

    for (const part of manifest.parts) {
      const partUrl = publicUrl(part.url);
      const geometry = this.visualGeometryCache.get(partUrl);
      if (!geometry) {
        continue;
      }
      const bodyId = this.bodyIdForName(part.bodyName);
      if (bodyId < 0) {
        throw new Error(`Visual mesh body not found: ${part.bodyName}`);
      }
      const key = part.key ?? `visual:${this.visualMeshInstances.length}:${part.meshName}`;
      const material = this.materialForRgba(part.rgba ?? [0.7, 0.7, 0.7, 1]);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.visible = this.visibleForMeshKey(key);
      this.visualMeshes.set(key, mesh);
      this.visualMeshGroup.add(mesh);

      const pos = part.pos ?? [0, 0, 0];
      const quat = part.quat ?? [1, 0, 0, 0];
      const localMatrix = new THREE.Matrix4().compose(
        new THREE.Vector3(pos[0] ?? 0, pos[1] ?? 0, pos[2] ?? 0),
        new THREE.Quaternion(quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0, quat[0] ?? 1).normalize(),
        new THREE.Vector3(1, 1, 1),
      );
      this.visualMeshInstances.push({ key, bodyId, mesh, localMatrix });

      this.renderParts.set(key, {
        key,
        label: part.label ?? this.partLabel(part.meshName, part.bodyName, ""),
        meshName: part.meshName,
        bodyName: part.bodyName,
        visible: true,
      });
      this.partVisibility.set(key, true);
    }
    this.updateVisualMeshes();
  }

  private updateVisualMeshes(): void {
    if (!this.data) {
      return;
    }
    const xpos = this.data.xpos;
    const xmat = this.data.xmat;
    for (const instance of this.visualMeshInstances) {
      const posOffset = instance.bodyId * 3;
      const matOffset = instance.bodyId * 9;
      this.bodyWorldMatrix.set(
        xmat[matOffset],
        xmat[matOffset + 1],
        xmat[matOffset + 2],
        xpos[posOffset],
        xmat[matOffset + 3],
        xmat[matOffset + 4],
        xmat[matOffset + 5],
        xpos[posOffset + 1],
        xmat[matOffset + 6],
        xmat[matOffset + 7],
        xmat[matOffset + 8],
        xpos[posOffset + 2],
        0,
        0,
        0,
        1,
      );
      instance.mesh.visible = this.visibleForMeshKey(instance.key);
      instance.mesh.matrix.copy(this.bodyWorldMatrix).multiply(instance.localMatrix);
      instance.mesh.matrixWorldNeedsUpdate = true;
    }
  }

  private firstMeshGeometry(root: THREE.Object3D): THREE.BufferGeometry {
    let geometry: THREE.BufferGeometry | null = null;
    root.traverse((object) => {
      const maybeMesh = object as THREE.Mesh;
      if (!geometry && maybeMesh.isMesh) {
        geometry = maybeMesh.geometry as THREE.BufferGeometry;
      }
    });
    if (!geometry) {
      throw new Error("GLB visual asset did not contain a mesh");
    }
    return geometry;
  }

  private buildSkeleton(): void {
    this.clearSkeleton();
    if (!this.model || !this.data) {
      return;
    }

    const robotBodies: number[] = [];
    for (let bodyId = 1; bodyId < this.model.nbody; bodyId += 1) {
      const bodyName = String(this.data.body(bodyId).name ?? "");
      if (bodyName.startsWith("robot/")) {
        robotBodies.push(bodyId);
      }
    }
    const robotBodySet = new Set(robotBodies);
    this.skeletonBodyIds = robotBodies;
    this.bodyPairs = robotBodies
      .map((bodyId) => [this.model.body_parentid[bodyId], bodyId] as [number, number])
      .filter(([parentId]) => robotBodySet.has(parentId));

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(this.bodyPairs.length * 6), 3),
    );
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0xffd166,
      depthTest: false,
      transparent: true,
      opacity: 0.95,
    });
    this.skeletonLines = new THREE.LineSegments(lineGeometry, lineMaterial);
    this.skeletonLines.frustumCulled = false;
    this.skeletonLines.renderOrder = 20;
    this.skeletonGroup.add(this.skeletonLines);

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(this.skeletonBodyIds.length * 3), 3),
    );
    const pointMaterial = new THREE.PointsMaterial({
      color: 0x64d2ff,
      depthTest: false,
      size: 0.045,
      sizeAttenuation: true,
    });
    this.skeletonPoints = new THREE.Points(pointGeometry, pointMaterial);
    this.skeletonPoints.frustumCulled = false;
    this.skeletonPoints.renderOrder = 21;
    this.skeletonGroup.add(this.skeletonPoints);
    this.updateSkeleton();
  }

  private updateSkeleton(): void {
    if (!this.data || !this.skeletonLines || !this.skeletonPoints) {
      return;
    }
    const xpos = this.data.xpos;
    const linePosition = this.skeletonLines.geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    const lineArray = linePosition.array as Float32Array;
    for (let i = 0; i < this.bodyPairs.length; i += 1) {
      const [parentId, bodyId] = this.bodyPairs[i];
      const target = i * 6;
      lineArray[target] = xpos[parentId * 3];
      lineArray[target + 1] = xpos[parentId * 3 + 1];
      lineArray[target + 2] = xpos[parentId * 3 + 2];
      lineArray[target + 3] = xpos[bodyId * 3];
      lineArray[target + 4] = xpos[bodyId * 3 + 1];
      lineArray[target + 5] = xpos[bodyId * 3 + 2];
    }
    linePosition.needsUpdate = true;

    const pointPosition = this.skeletonPoints.geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    const pointArray = pointPosition.array as Float32Array;
    for (let i = 0; i < this.skeletonBodyIds.length; i += 1) {
      const bodyId = this.skeletonBodyIds[i];
      const target = i * 3;
      pointArray[target] = xpos[bodyId * 3];
      pointArray[target + 1] = xpos[bodyId * 3 + 1];
      pointArray[target + 2] = xpos[bodyId * 3 + 2];
    }
    pointPosition.needsUpdate = true;
  }

  private meshKeyForGeom(index: number, mjvGeom: any): string {
    if (mjvGeom.objid >= 0) {
      return `${mjvGeom.objtype}:${mjvGeom.objid}`;
    }
    return `scene:${index}:${mjvGeom.type}:${mjvGeom.dataid}`;
  }

  private meshKeyForModelGeom(geomId: number): string {
    return `${enumValue(this.mujoco.mjtObj.mjOBJ_GEOM)}:${geomId}`;
  }

  private modelMeshIdForSceneGeom(mjvGeom: any): number {
    if (
      this.model &&
      mjvGeom.type === enumValue(this.mujoco.mjtGeom.mjGEOM_MESH) &&
      mjvGeom.objtype === enumValue(this.mujoco.mjtObj.mjOBJ_GEOM) &&
      mjvGeom.objid >= 0
    ) {
      return this.model.geom_dataid[mjvGeom.objid];
    }
    return mjvGeom.dataid;
  }

  private modelHfieldIdForSceneGeom(mjvGeom: any): number {
    if (
      this.model &&
      mjvGeom.type === enumValue(this.mujoco.mjtGeom.mjGEOM_HFIELD) &&
      mjvGeom.objtype === enumValue(this.mujoco.mjtObj.mjOBJ_GEOM) &&
      mjvGeom.objid >= 0
    ) {
      return this.model.geom_dataid[mjvGeom.objid];
    }
    return mjvGeom.dataid;
  }

  private visibleForMeshKey(meshKey: string, geomType?: number): boolean {
    if (!this.meshesEnabled || !(this.partVisibility.get(meshKey) ?? true)) {
      return false;
    }
    const type = geomType ?? this.meshes.get(meshKey)?.userData?.geomType;
    if (type === enumValue(this.mujoco.mjtGeom.mjGEOM_HFIELD)) {
      return this.heightfieldEnabled;
    }
    return true;
  }

  private objectName(objectType: unknown, id: number): string {
    const name = this.model ? this.mujoco.mj_id2name(this.model, enumValue(objectType), id) : "";
    return name || `<${id}>`;
  }

  private bodyIdForName(name: string): number {
    if (!this.model) {
      return -1;
    }
    for (let bodyId = 0; bodyId < this.model.nbody; bodyId += 1) {
      if (this.objectName(this.mujoco.mjtObj.mjOBJ_BODY, bodyId) === name) {
        return bodyId;
      }
    }
    return -1;
  }

  private partLabel(meshName: string, bodyName: string, geomName: string): string {
    const meshLeaf = stripRobotPrefix(meshName);
    const bodyLeaf = stripRobotPrefix(bodyName);
    if (meshLeaf === bodyLeaf) {
      return meshLeaf;
    }
    const geomLeaf = stripRobotPrefix(geomName);
    if (geomLeaf && !geomLeaf.startsWith("<") && geomLeaf !== meshLeaf) {
      return `${meshLeaf} @ ${geomLeaf}`;
    }
    return `${meshLeaf} @ ${bodyLeaf}`;
  }

  private meshForGeom(meshKey: string, mjvGeom: any): THREE.Mesh {
    const geometry = this.geometryForGeom(mjvGeom);
    const material = this.materialForGeom(mjvGeom);
    let mesh = this.meshes.get(meshKey);
    if (!mesh) {
      mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.visible = this.visibleForMeshKey(meshKey, mjvGeom.type);
      mesh.userData = {
        objtype: mjvGeom.objtype,
        objid: mjvGeom.objid,
        geomType: mjvGeom.type,
      };
      this.meshes.set(meshKey, mesh);
      this.dynamicGroup.add(mesh);
      return mesh;
    }
    if (mesh.geometry !== geometry) {
      mesh.geometry = geometry;
    }
    if (mesh.material !== material) {
      mesh.material = material;
    }
    if (!mesh.userData) {
      mesh.userData = {};
    }
    mesh.userData.objtype = mjvGeom.objtype;
    mesh.userData.objid = mjvGeom.objid;
    mesh.userData.geomType = mjvGeom.type;
    return mesh;
  }

  private geometryForGeom(mjvGeom: any): THREE.BufferGeometry {
    const meshId = this.modelMeshIdForSceneGeom(mjvGeom);
    const hfieldId = this.modelHfieldIdForSceneGeom(mjvGeom);
    const key = JSON.stringify([
      mjvGeom.type,
      Array.from(mjvGeom.size),
      meshId,
      hfieldId,
      this.env?.render?.mirrorMeshY ?? false,
    ]);
    const cached = this.geometryCache.get(key);
    if (cached) {
      return cached;
    }

    let geometry: THREE.BufferGeometry;
    if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_PLANE.value) {
      const width = mjvGeom.size[0] > 0 ? 2 * mjvGeom.size[0] : 80;
      const height = mjvGeom.size[1] > 0 ? 2 * mjvGeom.size[1] : 80;
      geometry = new THREE.PlaneGeometry(width, height);
    } else if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(mjvGeom.size[0], 32, 18);
    } else if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(mjvGeom.size[0], 2 * mjvGeom.size[2], 8, 18);
      geometry.rotateX(0.5 * Math.PI);
    } else if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(
        mjvGeom.size[0],
        mjvGeom.size[0],
        2 * mjvGeom.size[2],
        32,
      );
      geometry.rotateX(0.5 * Math.PI);
    } else if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(
        2 * mjvGeom.size[0],
        2 * mjvGeom.size[1],
        2 * mjvGeom.size[2],
      );
    } else if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1, 32, 18);
      geometry.scale(mjvGeom.size[0], mjvGeom.size[1], mjvGeom.size[2]);
    } else if (mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_MESH.value && meshId >= 0) {
      geometry = this.meshGeometry(meshId);
    } else if (mjvGeom.type === enumValue(this.mujoco.mjtGeom.mjGEOM_HFIELD) && hfieldId >= 0) {
      geometry = this.hfieldGeometry(hfieldId);
    } else {
      geometry = new THREE.BufferGeometry();
    }

    this.geometryCache.set(key, geometry);
    return geometry;
  }

  private meshGeometry(meshId: number): THREE.BufferGeometry {
    if (!this.model) {
      return new THREE.BufferGeometry();
    }
    const vertAdr = this.model.mesh_vertadr[meshId];
    const faceAdr = this.model.mesh_faceadr[meshId];
    const faceNum = this.model.mesh_facenum[meshId];
    const meshVert = this.model.mesh_vert;
    const meshFace = this.model.mesh_face;
    const positions = new Float32Array(faceNum * 9);
    const yScale = this.env?.render?.mirrorMeshY ? -1 : 1;

    for (let face = 0; face < faceNum; face += 1) {
      for (let corner = 0; corner < 3; corner += 1) {
        const vertexIndex = meshFace[(faceAdr + face) * 3 + corner];
        const source = (vertAdr + vertexIndex) * 3;
        const target = face * 9 + corner * 3;
        positions[target] = meshVert[source];
        positions[target + 1] = yScale * meshVert[source + 1];
        positions[target + 2] = meshVert[source + 2];
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeVertexNormals();
    return geometry;
  }

  private hfieldGeometry(hfieldId: number): THREE.BufferGeometry {
    if (!this.model || !this.model.hfield_data) {
      return new THREE.BufferGeometry();
    }
    const nrow = this.model.hfield_nrow?.[hfieldId] ?? 0;
    const ncol = this.model.hfield_ncol?.[hfieldId] ?? 0;
    const adr = this.model.hfield_adr?.[hfieldId] ?? -1;
    const sizeBase = hfieldId * 4;
    const sizeX = this.model.hfield_size?.[sizeBase] ?? 0;
    const sizeY = this.model.hfield_size?.[sizeBase + 1] ?? 0;
    const sizeZ = this.model.hfield_size?.[sizeBase + 2] ?? 0;
    if (nrow < 2 || ncol < 2 || adr < 0 || sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) {
      return new THREE.BufferGeometry();
    }

    const positions = new Float32Array((nrow - 1) * (ncol - 1) * 18);
    let target = 0;
    for (let row = 0; row < nrow - 1; row += 1) {
      for (let col = 0; col < ncol - 1; col += 1) {
        const v00 = this.hfieldVertex(adr, nrow, ncol, row, col, sizeX, sizeY, sizeZ);
        const v10 = this.hfieldVertex(adr, nrow, ncol, row, col + 1, sizeX, sizeY, sizeZ);
        const v01 = this.hfieldVertex(adr, nrow, ncol, row + 1, col, sizeX, sizeY, sizeZ);
        const v11 = this.hfieldVertex(adr, nrow, ncol, row + 1, col + 1, sizeX, sizeY, sizeZ);
        target = writeTriangle(positions, target, v00, v10, v11);
        target = writeTriangle(positions, target, v00, v11, v01);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeVertexNormals();
    return geometry;
  }

  private hfieldVertex(
    adr: number,
    nrow: number,
    ncol: number,
    row: number,
    col: number,
    sizeX: number,
    sizeY: number,
    sizeZ: number,
  ): [number, number, number] {
    return [
      -sizeX + (2 * sizeX * col) / (ncol - 1),
      -sizeY + (2 * sizeY * row) / (nrow - 1),
      this.model.hfield_data[adr + row * ncol + col] * sizeZ,
    ];
  }

  private materialForGeom(mjvGeom: any): THREE.Material {
    const alpha = mjvGeom.rgba[3];
    const hfieldWithSplatVisual =
      mjvGeom.type === enumValue(this.mujoco.mjtGeom.mjGEOM_HFIELD) &&
      Boolean(this.gaussianSplatMesh);
    const doubleSided =
      mjvGeom.type === this.mujoco.mjtGeom.mjGEOM_MESH.value ||
      mjvGeom.type === enumValue(this.mujoco.mjtGeom.mjGEOM_HFIELD);
    const key = [
      doubleSided ? "double" : "front",
      hfieldWithSplatVisual ? "gs-overlay" : "solid",
      ...Array.from(mjvGeom.rgba).map((value) => Number(value).toFixed(3)),
    ].join(",");
    const cached = this.materialCache.get(key);
    if (cached) {
      return cached;
    }
    const material = hfieldWithSplatVisual
      ? new THREE.MeshBasicMaterial({
          color: 0x35d7ff,
          transparent: true,
          opacity: 0.28,
          wireframe: true,
          side: THREE.DoubleSide,
          depthWrite: false,
          depthTest: true,
        })
      : new THREE.MeshStandardMaterial({
          color: new THREE.Color(mjvGeom.rgba[0], mjvGeom.rgba[1], mjvGeom.rgba[2]),
          roughness: 0.74,
          metalness: 0.08,
          side: doubleSided ? THREE.DoubleSide : THREE.FrontSide,
          opacity: alpha,
          transparent: alpha < 0.99,
          depthWrite: alpha > 0.5,
          depthTest: true,
        });
    this.materialCache.set(key, material);
    return material;
  }

  private async loadGaussianSplatVisual(
    visualSource: GaussianSplatVisualSource | null,
  ): Promise<void> {
    this.clearGaussianSplatVisual();
    if (!visualSource) {
      return;
    }

    try {
      const { PackedSplats, SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
      this.sparkRenderer = new SparkRenderer({ renderer: this.renderer });
      this.scene.add(this.sparkRenderer);

      const { mesh, matrixBaked } = await this.createGaussianSplatMesh(
        SplatMesh,
        PackedSplats,
        visualSource,
      );
      await mesh.initialized;
      if (matrixBaked) {
        mesh.matrixAutoUpdate = false;
        mesh.matrix.identity();
      } else {
        mesh.matrixAutoUpdate = true;
        applyObjectTransformFromMatrix(mesh, visualSource.matrix);
      }
      mesh.matrixWorldNeedsUpdate = true;
      mesh.updateMatrixWorld(true);
      mesh.frustumCulled = false;
      mesh.opacity = 1;
      mesh.visible = this.meshesEnabled;
      this.gaussianSplatMesh = mesh;
      this.gaussianSplatGroup.visible = this.meshesEnabled;
      this.scene.add(mesh);
      mesh.updateMatrixWorld(true);
    } catch (error) {
      console.warn("Failed to render Gaussian splat source", error);
      this.clearGaussianSplatVisual();
    }
  }

  private async createGaussianSplatMesh(
    SplatMesh: any,
    PackedSplats: any,
    visualSource: GaussianSplatVisualSource,
  ): Promise<{ mesh: any; matrixBaked: boolean }> {
    if (matrixDeterminant3x3(visualSource.matrix) >= 0) {
      return {
        mesh: new SplatMesh({
          fileBytes: visualSource.source.bytes,
          fileName: visualSource.source.name,
          extSplats: !visualSource.source.name.toLowerCase().endsWith(".splat"),
        }),
        matrixBaked: false,
      };
    }

    const sourceMesh = new SplatMesh({
      fileBytes: visualSource.source.bytes,
      fileName: visualSource.source.name,
      extSplats: !visualSource.source.name.toLowerCase().endsWith(".splat"),
    });
    await sourceMesh.initialized;
    try {
      const packedSplats = new PackedSplats({ maxSplats: sourceMesh.numSplats });
      packedSplats.ensureSplats(sourceMesh.numSplats);
      const matrix = matrixFromArray(visualSource.matrix);
      const scaleFactor = averageColumnScale(visualSource.matrix);
      const centerOut = new THREE.Vector3();
      const scalesOut = new THREE.Vector3();

      sourceMesh.forEachSplat((
        index: number,
        center: THREE.Vector3,
        scales: THREE.Vector3,
        quaternion: THREE.Quaternion,
        opacity: number,
        color: THREE.Color,
      ) => {
        centerOut.copy(center).applyMatrix4(matrix);
        scalesOut.copy(scales).multiplyScalar(scaleFactor);
        packedSplats.setSplat(index, centerOut, scalesOut, quaternion, opacity, color);
      });

      return {
        mesh: new SplatMesh({
          packedSplats,
          editable: false,
          enableLod: false,
          raycastable: false,
        }),
        matrixBaked: true,
      };
    } finally {
      sourceMesh.dispose();
    }
  }

  private materialForRgba(rgba: number[]): THREE.Material {
    const alpha = rgba[3] ?? 1;
    const key = [
      "visual",
      ...rgba.map((value) => Number(value).toFixed(3)),
    ].join(",");
    const cached = this.materialCache.get(key);
    if (cached) {
      return cached;
    }
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0] ?? 0.7, rgba[1] ?? 0.7, rgba[2] ?? 0.7),
      roughness: 0.74,
      metalness: 0.08,
      side: THREE.DoubleSide,
      opacity: alpha,
      transparent: alpha < 0.99,
      depthWrite: alpha > 0.5,
    });
    this.materialCache.set(key, material);
    return material;
  }

  private resize(): void {
    const width = Math.max(1, this.host.clientWidth);
    const height = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private clearDynamicMeshes(): void {
    this.clearSpawnPickerMarker();
    this.clearRouteGoalMarker();
    for (const mesh of this.meshes.values()) {
      this.dynamicGroup.remove(mesh);
    }
    this.meshes.clear();
    for (const mesh of this.visualMeshes.values()) {
      this.visualMeshGroup.remove(mesh);
    }
    this.visualMeshes.clear();
    this.visualMeshInstances = [];
    for (const geometry of this.geometryCache.values()) {
      geometry.dispose();
    }
    this.geometryCache.clear();
    for (const geometry of this.visualGeometryCache.values()) {
      geometry.dispose();
    }
    this.visualGeometryCache.clear();
    for (const material of this.materialCache.values()) {
      material.dispose();
    }
    this.materialCache.clear();
    this.renderParts.clear();
    this.partVisibility.clear();
    this.clearGaussianSplatVisual();
    this.clearSkeleton();
    this.clearRouteLine();
  }

  private clearSpawnPickerMarker(): void {
    this.spawnPickerEnabled = false;
    if (!this.spawnPickerMarker) {
      return;
    }
    this.scene.remove(this.spawnPickerMarker);
    this.spawnPickerMarker.geometry.dispose();
    disposeMaterial(this.spawnPickerMarker.material);
    this.spawnPickerMarker = null;
  }

  private clearRouteGoalMarker(): void {
    if (!this.routeGoalMarker) {
      return;
    }
    this.scene.remove(this.routeGoalMarker);
    this.routeGoalMarker.geometry.dispose();
    disposeMaterial(this.routeGoalMarker.material);
    this.routeGoalMarker = null;
  }

  private clearGaussianSplatVisual(): void {
    if (this.gaussianSplatMesh) {
      this.gaussianSplatGroup.remove(this.gaussianSplatMesh);
      this.scene.remove(this.gaussianSplatMesh);
      this.gaussianSplatMesh.dispose?.();
      this.gaussianSplatMesh = null;
    }
    this.gaussianSplatGroup.clear();
    if (this.sparkRenderer) {
      this.scene.remove(this.sparkRenderer);
      this.sparkRenderer.dispose?.();
      this.sparkRenderer = null;
    }
  }

  private clearSkeleton(): void {
    this.skeletonGroup.clear();
    this.bodyPairs = [];
    this.skeletonBodyIds = [];
    if (this.skeletonLines) {
      this.skeletonLines.geometry.dispose();
      disposeMaterial(this.skeletonLines.material);
      this.skeletonLines = null;
    }
    if (this.skeletonPoints) {
      this.skeletonPoints.geometry.dispose();
      disposeMaterial(this.skeletonPoints.material);
      this.skeletonPoints = null;
    }
  }

  private clearRouteLine(): void {
    if (!this.routeLine) {
      return;
    }
    this.routeLineGroup.remove(this.routeLine);
    this.routeLine.geometry.dispose();
    disposeMaterial(this.routeLine.material);
    this.routeLine = null;
  }

  private disposeMujocoScene(): void {
    this.mjvScene?.delete();
    this.mjvCamera?.delete();
    this.mjvPerturb?.delete();
    this.mjvOption?.delete();
    this.mjvScene = null;
    this.mjvCamera = null;
    this.mjvPerturb = null;
    this.mjvOption = null;
  }
}

function disposeMaterial(material: THREE.Material | THREE.Material[]): void {
  if (Array.isArray(material)) {
    for (const item of material) {
      item.dispose();
    }
    return;
  }
  material.dispose();
}

function enumValue(value: any): number {
  return typeof value === "object" && value !== null ? value.value : value;
}

function writeTriangle(
  positions: Float32Array,
  target: number,
  a: [number, number, number],
  b: [number, number, number],
  c: [number, number, number],
): number {
  positions[target] = a[0];
  positions[target + 1] = a[1];
  positions[target + 2] = a[2];
  positions[target + 3] = b[0];
  positions[target + 4] = b[1];
  positions[target + 5] = b[2];
  positions[target + 6] = c[0];
  positions[target + 7] = c[1];
  positions[target + 8] = c[2];
  return target + 9;
}


function roundUp(value: number, step: number): number {
  return Math.ceil(value / step) * step;
}

function applyObjectTransformFromMatrix(object: THREE.Object3D, matrix: number[]): void {
  const transform = matrixFromArray(matrix);
  transform.decompose(object.position, object.quaternion, object.scale);
}

function matrixFromArray(matrix: number[]): THREE.Matrix4 {
  return new THREE.Matrix4().set(
    matrix[0], matrix[1], matrix[2], matrix[3],
    matrix[4], matrix[5], matrix[6], matrix[7],
    matrix[8], matrix[9], matrix[10], matrix[11],
    matrix[12], matrix[13], matrix[14], matrix[15],
  );
}

function matrixDeterminant3x3(matrix: number[]): number {
  return (
    matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9]) -
    matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8]) +
    matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8])
  );
}

function averageColumnScale(matrix: number[]): number {
  const sx = Math.hypot(matrix[0], matrix[4], matrix[8]);
  const sy = Math.hypot(matrix[1], matrix[5], matrix[9]);
  const sz = Math.hypot(matrix[2], matrix[6], matrix[10]);
  return (sx + sy + sz) / 3;
}

function stripRobotPrefix(value: string): string {
  return value.replace(/^robot\//, "");
}
