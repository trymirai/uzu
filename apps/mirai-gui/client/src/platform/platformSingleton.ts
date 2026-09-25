import type { PlatformClient } from "./PlatformClient";

let client: PlatformClient | null = null;

export function initPlatform(impl: PlatformClient): void {
  if (client !== null) {
    throw new Error("Platform already initialized.");
  }
  client = impl;
}

export function getPlatform(): PlatformClient {
  if (client === null) {
    throw new Error("Platform not initialized. Call initPlatform() before use.");
  }
  return client;
}
