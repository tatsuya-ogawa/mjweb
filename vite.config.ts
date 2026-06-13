import { defineConfig } from "vite";

const base = normalizeBase(process.env.MJWEB_BASE_PATH);

export default defineConfig({
  base,
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
    fs: {
      allow: ["."],
    },
  },
});

function normalizeBase(value: string | undefined): string {
  if (!value) {
    return "/";
  }
  if (/^https?:\/\//i.test(value)) {
    return value.endsWith("/") ? value : `${value}/`;
  }
  const withLeadingSlash = value.startsWith("/") ? value : `/${value}`;
  return withLeadingSlash.endsWith("/") ? withLeadingSlash : `${withLeadingSlash}/`;
}
