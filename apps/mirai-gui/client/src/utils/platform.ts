export function isMacPlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  const platform = navigator.platform ?? "";
  if (platform) return platform.toLowerCase().includes("mac");
  return (navigator.userAgent ?? "").includes("Mac");
}
