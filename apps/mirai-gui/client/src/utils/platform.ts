export function isMacPlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  const platform = navigator.platform ?? "";
  if (platform) return platform.toLowerCase().includes("mac");
  return (navigator.userAgent ?? "").includes("Mac");
}

export function isWindowsPlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  const platform = navigator.platform ?? "";
  if (platform) return platform.toLowerCase().startsWith("win");
  return (navigator.userAgent ?? "").includes("Windows");
}
