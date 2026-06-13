import { g1BackflipEnv } from "./g1Backflip";
import { g1FlatEnv } from "./g1Flat";
import { g1RoughEnv } from "./g1Rough";
import { go1FlatEnv } from "./go1Flat";
import { go1GaussianEnv } from "./go1Gaussian";
import { go1RoughEnv } from "./go1Rough";
import type { EnvDefinition } from "./types";

export const envRegistry: EnvDefinition[] = [
  g1FlatEnv,
  g1RoughEnv,
  go1FlatEnv,
  go1RoughEnv,
  go1GaussianEnv,
];

// Envs awaiting full implementation (e.g. ONNX or pipeline still pending).
export const plannedEnvRegistry: EnvDefinition[] = [g1BackflipEnv];

export const allEnvRegistry: EnvDefinition[] = [...envRegistry, ...plannedEnvRegistry];

export function findEnvDefinition(id: string): EnvDefinition {
  const env = allEnvRegistry.find((item) => item.id === id);
  if (!env) {
    throw new Error(`Unknown environment: ${id}`);
  }
  return env;
}
