import { bundledGaussianSourceUrls } from "../generated/gaussianSources";

interface BundledGaussianSource {
  envId: string;
  bundleId: string;
  url: string;
  fileName: string;
}

export interface GaussianSourceBundleDefinition {
  id: string;
  label: string;
  sourceUrls: string[];
}

const SOURCE_EXTENSION_PRIORITY = new Map([
  [".sog", 0],
  [".spz", 1],
  [".splat", 2],
  [".ply", 3],
]);

const bundledGaussianSources = bundledGaussianSourceUrls.map(
  (url): BundledGaussianSource => {
    const parsed = parseGaussianSourceUrl(url);
    return {
      envId: parsed.envId,
      bundleId: parsed.bundleId,
      url,
      fileName: url.split("/").pop() ?? url,
    };
  },
);

export function gaussianSourceBundlesForEnv(envId: string): GaussianSourceBundleDefinition[] {
  const bundles = new Map<string, BundledGaussianSource[]>();
  for (const source of bundledGaussianSources) {
    if (source.envId !== envId) {
      continue;
    }
    const sources = bundles.get(source.bundleId) ?? [];
    sources.push(source);
    bundles.set(source.bundleId, sources);
  }
  return [...bundles.entries()]
    .map(([id, sources]) => ({
      id,
      label: formatBundleLabel(id),
      sourceUrls: sortBundledGaussianSources(sources).map((source) => source.url),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

export function gaussianSourceUrlsForBundle(
  envId: string | undefined,
  bundleId: string | undefined,
): string[] {
  if (!envId || !bundleId) {
    return [];
  }
  return bundledGaussianSources
    .filter((source) => source.envId === envId && source.bundleId === bundleId)
    .sort(compareBundledGaussianSources)
    .map((source) => source.url);
}

function sortBundledGaussianSources(sources: BundledGaussianSource[]): BundledGaussianSource[] {
  return [...sources].sort(compareBundledGaussianSources);
}

function parseGaussianSourceUrl(url: string): { envId: string; bundleId: string } {
  const parts = url.split("/").filter(Boolean);
  const envIndex = parts.indexOf("envs");
  const splatsIndex = parts.indexOf("splats");
  const envId = envIndex >= 0 ? parts[envIndex + 1] : "";
  const afterSplats = splatsIndex >= 0 ? parts.slice(splatsIndex + 1) : [];
  const fileName = afterSplats[afterSplats.length - 1] ?? "";
  const bundleId = afterSplats.length > 1
    ? afterSplats[0]
    : fileName.replace(/\.[^.]+$/, "");
  if (!envId || !bundleId) {
    throw new Error(`Invalid Gaussian source URL: ${url}`);
  }
  return { envId, bundleId };
}

function formatBundleLabel(id: string): string {
  return id
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function compareBundledGaussianSources(
  a: BundledGaussianSource,
  b: BundledGaussianSource,
): number {
  const extPriority = extensionPriority(a.fileName) - extensionPriority(b.fileName);
  return extPriority || a.fileName.localeCompare(b.fileName);
}

function extensionPriority(fileName: string): number {
  const extension = fileName.slice(fileName.lastIndexOf(".")).toLowerCase();
  return SOURCE_EXTENSION_PRIORITY.get(extension) ?? Number.MAX_SAFE_INTEGER;
}
