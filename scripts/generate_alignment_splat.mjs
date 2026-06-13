import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { deflateSync } from "node:zlib";

const outDir = resolve("public/envs/go1_gaussian/splats/cardinal");
const outPath = resolve(outDir, "cardinal_towers.splat");
const sogOutPath = resolve(outDir, "cardinal_towers.sog");
const splats = [];

const GRID_STEP = 0.25;
const DISC_STEP = 0.08;
const TOWER_VERTICAL_STEP = 0.1;

function addSplat(x, y, z, color, scale = 0.08, opacity = 235) {
  splats.push({ x, y, z, color, scale, opacity });
}

function addLine(a, b, color, count, scale = 0.07, opacity = 235) {
  for (let i = 0; i < count; i += 1) {
    const t = count === 1 ? 0 : i / (count - 1);
    addSplat(
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
      color,
      scale,
      opacity,
    );
  }
}

function addLineBySpacing(a, b, color, spacing, scale = 0.07, opacity = 235) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  const length = Math.hypot(dx, dy, dz);
  addLine(a, b, color, Math.max(2, Math.round(length / spacing) + 1), scale, opacity);
}

function addDisc(cx, cy, cz, radius, color, step = 0.2, scale = 0.055, opacity = 175) {
  for (let x = -radius; x <= radius + 1e-6; x += step) {
    for (let z = -radius; z <= radius + 1e-6; z += step) {
      if (x * x + z * z <= radius * radius) {
        addSplat(cx + x, cy, cz + z, color, scale, opacity);
      }
    }
  }
}

function addTower(cx, cz, height, color) {
  addDisc(cx, 0, cz, 0.55, color, DISC_STEP, 0.055, 225);
  for (let y = 0; y <= height + 1e-6; y += TOWER_VERTICAL_STEP) {
    addDisc(cx, y, cz, 0.32, color, DISC_STEP, 0.06, 245);
  }
  addDisc(cx, height, cz, 0.52, [245, 245, 245], DISC_STEP, 0.05, 240);
}

const gray = [120, 124, 132];
const white = [245, 245, 245];
const eastRed = [255, 48, 48];
const northGreen = [40, 220, 95];
const westBlue = [70, 135, 255];
const southYellow = [255, 210, 55];

const DENSE_GRID_STEP = 0.08;
for (let x = -5; x <= 5.001; x += DENSE_GRID_STEP) {
  for (let z = -5; z <= 5.001; z += DENSE_GRID_STEP) {
    addSplat(x, 0, z, gray, 0.035, 210);
  }
}

addDisc(0, 0.04, 0, 0.75, white, DISC_STEP, 0.055, 230);

addLineBySpacing([0, 0.08, 0], [4, 0.08, 0], eastRed, 0.05, 0.055);
addLineBySpacing([0, 0.1, 0], [0, 0.1, 4], northGreen, 0.05, 0.055);
addLineBySpacing([0, 0.12, 0], [-4, 0.12, 0], westBlue, 0.05, 0.055);
addLineBySpacing([0, 0.14, 0], [0, 0.14, -4], southYellow, 0.05, 0.055);

// Source coordinates are x=east/west, y=height, z=north/south.
// The four towers intentionally have very different heights to expose vertical and directional offsets.
addTower(4, 0, 0.75, eastRed);    // East: 0.75 m
addTower(0, 4, 2.5, northGreen);  // North: 2.5 m
addTower(-4, 0, 5, westBlue);     // West: 5 m
addTower(0, -4, 9, southYellow);  // South: 9 m

const buffer = Buffer.alloc(splats.length * 32);
for (let i = 0; i < splats.length; i += 1) {
  const splat = splats[i];
  const offset = i * 32;
  buffer.writeFloatLE(splat.x, offset);
  buffer.writeFloatLE(splat.y, offset + 4);
  buffer.writeFloatLE(splat.z, offset + 8);
  buffer.writeFloatLE(splat.scale, offset + 12);
  buffer.writeFloatLE(splat.scale, offset + 16);
  buffer.writeFloatLE(splat.scale, offset + 20);
  buffer[offset + 24] = splat.color[0];
  buffer[offset + 25] = splat.color[1];
  buffer[offset + 26] = splat.color[2];
  buffer[offset + 27] = splat.opacity;
  buffer[offset + 28] = 255;
  buffer[offset + 29] = 128;
  buffer[offset + 30] = 128;
  buffer[offset + 31] = 128;
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, buffer);

const sogBytes = encodePcSogsZip(splats.map(toSogSourceSplat));
writeFileSync(sogOutPath, sogBytes);

console.log(`Wrote ${splats.length} splats to ${outPath}`);
console.log(`Wrote ${splats.length} PC SOGS splats to ${sogOutPath}`);

function toSogSourceSplat(splat) {
  return {
    ...splat,
    // Match the COLMAP-style SOG convention used by the real samples:
    // +Y is downward; the app applies the TPS-style X rotation that flips Y and Z.
    y: -splat.y,
  };
}

function encodePcSogsZip(sourceSplats) {
  const { width, height, pixels } = imageLayout(sourceSplats.length);
  const means = encodeMeans(sourceSplats, width, height, pixels);
  const scales = encodeScales(sourceSplats, width, height, pixels);
  const sh0 = encodeSh0(sourceSplats, width, height, pixels);
  const quats = encodeQuats(sourceSplats.length, width, height, pixels);
  const meta = {
    version: 2,
    count: sourceSplats.length,
    antialias: false,
    means: {
      mins: means.mins,
      maxs: means.maxs,
      files: ["means_l.png", "means_u.png"],
    },
    scales: {
      codebook: scales.codebook,
      files: ["scales.png"],
    },
    quats: {
      files: ["quats.png"],
    },
    sh0: {
      codebook: sh0.codebook,
      files: ["sh0.png"],
    },
  };

  return zipStore({
    "meta.json": Buffer.from(JSON.stringify(meta, null, 2)),
    "means_l.png": pngRgba(width, height, means.low),
    "means_u.png": pngRgba(width, height, means.high),
    "scales.png": pngRgba(width, height, scales.rgba),
    "quats.png": pngRgba(width, height, quats),
    "sh0.png": pngRgba(width, height, sh0.rgba),
  });
}

function zipStore(files) {
  const localParts = [];
  const centralParts = [];
  let offset = 0;
  const dosTime = 0;
  const dosDate = ((2026 - 1980) << 9) | (1 << 5) | 1;
  const utf8Flag = 0x0800;

  for (const [name, fileData] of Object.entries(files)) {
    const data = Buffer.from(fileData);
    const nameBytes = Buffer.from(name, "utf8");
    const crc = crc32(data);
    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0);
    localHeader.writeUInt16LE(20, 4);
    localHeader.writeUInt16LE(utf8Flag, 6);
    localHeader.writeUInt16LE(0, 8);
    localHeader.writeUInt16LE(dosTime, 10);
    localHeader.writeUInt16LE(dosDate, 12);
    localHeader.writeUInt32LE(crc, 14);
    localHeader.writeUInt32LE(data.length, 18);
    localHeader.writeUInt32LE(data.length, 22);
    localHeader.writeUInt16LE(nameBytes.length, 26);
    localHeader.writeUInt16LE(0, 28);

    localParts.push(localHeader, nameBytes, data);

    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4);
    centralHeader.writeUInt16LE(20, 6);
    centralHeader.writeUInt16LE(utf8Flag, 8);
    centralHeader.writeUInt16LE(0, 10);
    centralHeader.writeUInt16LE(dosTime, 12);
    centralHeader.writeUInt16LE(dosDate, 14);
    centralHeader.writeUInt32LE(crc, 16);
    centralHeader.writeUInt32LE(data.length, 20);
    centralHeader.writeUInt32LE(data.length, 24);
    centralHeader.writeUInt16LE(nameBytes.length, 28);
    centralHeader.writeUInt16LE(0, 30);
    centralHeader.writeUInt16LE(0, 32);
    centralHeader.writeUInt16LE(0, 34);
    centralHeader.writeUInt16LE(0, 36);
    centralHeader.writeUInt32LE(0, 38);
    centralHeader.writeUInt32LE(offset, 42);
    centralParts.push(centralHeader, nameBytes);

    offset += localHeader.length + nameBytes.length + data.length;
  }

  const centralOffset = offset;
  const centralDirectory = Buffer.concat(centralParts);
  const end = Buffer.alloc(22);
  const entryCount = Object.keys(files).length;
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entryCount, 8);
  end.writeUInt16LE(entryCount, 10);
  end.writeUInt32LE(centralDirectory.length, 12);
  end.writeUInt32LE(centralOffset, 16);
  end.writeUInt16LE(0, 20);

  return Buffer.concat([...localParts, centralDirectory, end]);
}

function imageLayout(count) {
  const width = Math.ceil(Math.sqrt(count));
  const height = Math.ceil(count / width);
  return {
    width,
    height,
    pixels: width * height,
  };
}

function encodeMeans(sourceSplats, width, height, pixels) {
  const encodedCoords = sourceSplats.map((splat) => [
    signedLog1p(splat.x),
    signedLog1p(splat.y),
    signedLog1p(splat.z),
  ]);
  const mins = [0, 1, 2].map((axis) => Math.min(...encodedCoords.map((coord) => coord[axis])));
  const maxs = [0, 1, 2].map((axis) => Math.max(...encodedCoords.map((coord) => coord[axis])));
  const low = new Uint8Array(pixels * 4);
  const high = new Uint8Array(pixels * 4);

  for (let i = 0; i < sourceSplats.length; i += 1) {
    const offset = i * 4;
    for (let axis = 0; axis < 3; axis += 1) {
      const quantized = quantize16(encodedCoords[i][axis], mins[axis], maxs[axis]);
      low[offset + axis] = quantized & 0xff;
      high[offset + axis] = quantized >> 8;
    }
    low[offset + 3] = 255;
    high[offset + 3] = 255;
  }

  return { mins, maxs, low, high };
}

function encodeScales(sourceSplats, _width, _height, pixels) {
  const logScales = sourceSplats.map((splat) => Math.log(Math.max(1e-6, splat.scale)));
  const min = Math.min(...logScales);
  const max = Math.max(...logScales);
  const codebook = Array.from({ length: 256 }, (_, i) => interpolate(min, max, i / 255));
  const rgba = new Uint8Array(pixels * 4);

  for (let i = 0; i < sourceSplats.length; i += 1) {
    const scaleIndex = quantize8(logScales[i], min, max);
    const offset = i * 4;
    rgba[offset] = scaleIndex;
    rgba[offset + 1] = scaleIndex;
    rgba[offset + 2] = scaleIndex;
    rgba[offset + 3] = 255;
  }

  return { codebook, rgba };
}

function encodeQuats(count, _width, _height, pixels) {
  const rgba = new Uint8Array(pixels * 4);
  for (let i = 0; i < count; i += 1) {
    const offset = i * 4;
    rgba[offset] = 128;
    rgba[offset + 1] = 128;
    rgba[offset + 2] = 128;
    rgba[offset + 3] = 252;
  }
  return rgba;
}

function encodeSh0(sourceSplats, _width, _height, pixels) {
  const shC0 = 0.28209479177387814;
  const codebook = Array.from({ length: 256 }, (_, i) => (i / 255 - 0.5) / shC0);
  const rgba = new Uint8Array(pixels * 4);

  for (let i = 0; i < sourceSplats.length; i += 1) {
    const offset = i * 4;
    rgba[offset] = sourceSplats[i].color[0];
    rgba[offset + 1] = sourceSplats[i].color[1];
    rgba[offset + 2] = sourceSplats[i].color[2];
    rgba[offset + 3] = sourceSplats[i].opacity;
  }

  return { codebook, rgba };
}

function signedLog1p(value) {
  return Math.sign(value) * Math.log1p(Math.abs(value));
}

function interpolate(min, max, t) {
  return min + (max - min) * t;
}

function quantize8(value, min, max) {
  const range = max - min;
  if (Math.abs(range) < 1e-12) {
    return 0;
  }
  return Math.min(255, Math.max(0, Math.round(((value - min) / range) * 255)));
}

function quantize16(value, min, max) {
  const range = max - min;
  if (Math.abs(range) < 1e-12) {
    return 0;
  }
  return Math.min(65535, Math.max(0, Math.round(((value - min) / range) * 65535)));
}

function pngRgba(width, height, rgba) {
  const rowStride = width * 4;
  const raw = Buffer.alloc((rowStride + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const rawOffset = y * (rowStride + 1);
    raw[rawOffset] = 0;
    raw.set(rgba.subarray(y * rowStride, (y + 1) * rowStride), rawOffset + 1);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr(width, height)),
    pngChunk("IDAT", deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function ihdr(width, height) {
  const buffer = Buffer.alloc(13);
  buffer.writeUInt32BE(width, 0);
  buffer.writeUInt32BE(height, 4);
  buffer[8] = 8;
  buffer[9] = 6;
  buffer[10] = 0;
  buffer[11] = 0;
  buffer[12] = 0;
  return buffer;
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, crc]);
}

function crc32(bytes) {
  const table = crcTableForPng();
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function crcTableForPng() {
  if (!crcTableForPng.table) {
    crcTableForPng.table = Array.from({ length: 256 }, (_, index) => {
      let value = index;
      for (let bit = 0; bit < 8; bit += 1) {
        value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
      }
      return value >>> 0;
    });
  }
  return crcTableForPng.table;
}
