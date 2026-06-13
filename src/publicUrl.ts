export function publicUrl(path: string): string {
  if (isAbsoluteUrl(path)) {
    return path;
  }
  const base = import.meta.env.BASE_URL || "/";
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  const normalizedPath = path.replace(/^\/+/, "");
  return `${normalizedBase}${normalizedPath}`;
}

function isAbsoluteUrl(path: string): boolean {
  return /^(?:[a-z][a-z\d+.-]*:)?\/\//i.test(path) ||
    path.startsWith("data:") ||
    path.startsWith("blob:");
}
