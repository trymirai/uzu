import type { Engine } from "@uzu/wasm";

export type UzuBindings = typeof import("@uzu/wasm");
export type UzuRuntime = { api: UzuBindings; engine: Engine };

let initializing: Promise<UzuRuntime> | undefined;

// OPFS in kiban currently uses window.navigator, so initialize on the page.
export function getUzuRuntime(): Promise<UzuRuntime> {
  initializing ??= (async () => {
    const api = await import("@uzu/wasm");
    await api.default();
    const config = api.EngineConfig.create();
    config.downloadManagerType = api.DownloadManagerType.Universal;
    try {
      return { api, engine: await api.Engine.create(config) };
    } finally {
      config.free();
    }
  })().catch((cause: unknown) => {
    initializing = undefined;
    throw cause;
  });
  return initializing;
}
