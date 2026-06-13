import { gaussianSourceBundlesForEnv } from "./gaussianSourceBundles";
import { go1RoughEnv } from "./go1Rough";
import { publicUrl } from "../publicUrl";
import type {
  EnvDefinition,
  GaussianSplatHeightfieldConfig,
  GaussianSplatPresetDefinition,
} from "./types";

const roughPolicy = go1RoughEnv.policy;
const GO1_GAUSSIAN_ENV_ID = "go1_gaussian";

if (!roughPolicy) {
  throw new Error("Go1 rough policy is required for Gaussian support surface tasks");
}

const baseHeightfield: GaussianSplatHeightfieldConfig = {
  kind: "gaussian-splat",
  nrow: 49,
  ncol: 145,
  sizeX: 6,
  sizeY: 6,
  elevationScale: 3.4,
  baseThickness: 0.08,
  sourceCenter: "origin",
  sourceScaleMode: "metric",
  sourceMetersPerUnit: 1,
  sourceScaleMultiplier: 1,
  sourceBoundsPadding: 1,
  sourceTargetCellSize: 0.25,
  sourceMaxRows: 96,
  sourceMaxCols: 96,
  startAtGaussianOrigin: true,
  startPlacement: "support",
  smoothingPasses: 1,
  startHeightSampleRadius: 0.35,
  startX: 0,
  startY: 0,
  robotBaseHeight: 0.278,
};

const defaultViewer = {
  followBody: "robot/trunk",
  distance: 11,
  azimuthDeg: 135,
  elevationDeg: -30,
};

const presetOverrides: Record<string, {
  label?: string;
  taskId?: string;
  heightfield?: Partial<GaussianSplatHeightfieldConfig>;
  viewer?: EnvDefinition["viewer"];
}> = {
  cardinal: {
    label: "Cardinal",
    taskId: "Mjweb-Gaussian-Support-Cardinal-Unitree-Go1",
    heightfield: {
      nrow: 97,
      ncol: 193,
      sourceTargetCellSize: 0.125,
      sourceMaxRows: 160,
      sourceMaxCols: 256,
      supportChunkResolution: 96,
      supportFillMode: "nearby",
      supportFillIterations: 8,
      supportFillMaxHeightRange: 0.45,
    },
    viewer: defaultViewer,
  },
  cochem: {
    label: "Cochem",
    taskId: "Mjweb-Gaussian-Support-Cochem-Unitree-Go1",
    heightfield: {
      sizeX: 9,
      sizeY: 3,
      elevationScale: 1.2,
      sourceCenter: "density",
      sourceMaxRows: 512,
      sourceMaxCols: 1024,
      dynamicWindow: true,
      dynamicWindowUpdateDistance: 2,
      startAtGaussianOrigin: false,
      smoothingPasses: 5,
      startHeightSampleRadius: 0.55,
      startX: -6.6,
      startY: 0,
    },
    viewer: {
      followBody: "robot/trunk",
      distance: 3.2,
      azimuthDeg: 135,
      elevationDeg: -18,
    },
  },
};

const gaussianPresets: GaussianSplatPresetDefinition[] =
  gaussianSourceBundlesForEnv(GO1_GAUSSIAN_ENV_ID).map((bundle) => {
    const override = presetOverrides[bundle.id] ?? {};
    return {
      id: bundle.id,
      label: override.label ?? bundle.label,
      taskId: override.taskId ?? `Mjweb-Gaussian-Support-${bundle.id}-Unitree-Go1`,
      heightfield: {
        ...baseHeightfield,
        ...override.heightfield,
        sourceBundleId: bundle.id,
      },
      viewer: override.viewer ?? defaultViewer,
    };
  });

const defaultGaussianPreset = gaussianPresets.find((preset) => preset.id === "cardinal") ??
  gaussianPresets[0];

if (!defaultGaussianPreset) {
  throw new Error(`No Gaussian splat bundles found for ${GO1_GAUSSIAN_ENV_ID}`);
}

export const go1GaussianEnv: EnvDefinition = {
  ...go1RoughEnv,
  id: GO1_GAUSSIAN_ENV_ID,
  label: "Unitree Go1 Gaussian Support",
  taskId: defaultGaussianPreset.taskId ?? "Mjweb-Gaussian-Support-Unitree-Go1",
  sceneXmlUrl: publicUrl("envs/go1_flat/scene_optimized.xml"),
  observationKind: "velocity-rough-v1",
  policy: {
    ...roughPolicy,
    commandDefaults: {
      linVelX: 0,
      linVelY: 0,
      angVelZ: 0,
    },
    terrainScan: roughPolicy.terrainScan
      ? {
          ...roughPolicy.terrainScan,
          maxDistance: 6.0,
        }
      : undefined,
  },
  render: {
    visualMeshManifestUrl: publicUrl("envs/go1_flat/render-manifest.json"),
  },
  heightfield: defaultGaussianPreset.heightfield,
  gaussianPresets,
  viewer: defaultGaussianPreset.viewer ?? go1RoughEnv.viewer,
};
