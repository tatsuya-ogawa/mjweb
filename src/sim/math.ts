export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function quatRotateInverse(
  quatWxyz: ArrayLike<number>,
  vector: readonly [number, number, number],
): [number, number, number] {
  const w = quatWxyz[0];
  const x = -quatWxyz[1];
  const y = -quatWxyz[2];
  const z = -quatWxyz[3];
  const vx = vector[0];
  const vy = vector[1];
  const vz = vector[2];

  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);

  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

export function quatRotate(
  quatWxyz: ArrayLike<number>,
  vector: readonly [number, number, number],
): [number, number, number] {
  const w = quatWxyz[0];
  const x = quatWxyz[1];
  const y = quatWxyz[2];
  const z = quatWxyz[3];
  const vx = vector[0];
  const vy = vector[1];
  const vz = vector[2];

  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);

  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

export function yawFromQuat(quatWxyz: ArrayLike<number>): number {
  const w = quatWxyz[0];
  const x = quatWxyz[1];
  const y = quatWxyz[2];
  const z = quatWxyz[3];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

export function wrapPi(value: number): number {
  let wrapped = (value + Math.PI) % (2 * Math.PI);
  if (wrapped < 0) {
    wrapped += 2 * Math.PI;
  }
  return wrapped - Math.PI;
}

export function vectorNorm(values: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i] * values[i];
  }
  return Math.sqrt(total);
}
