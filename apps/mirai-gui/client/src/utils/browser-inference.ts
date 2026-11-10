/** Why this browser cannot run models, or null when it can. */
export function browserInferenceError(): string | null {
  if (!globalThis.isSecureContext) return "Running models in the browser requires HTTPS or localhost.";
  if (typeof navigator === "undefined" || !("gpu" in navigator))
    return "WebGPU is unavailable. Use a browser and device with WebGPU enabled.";
  return null;
}
