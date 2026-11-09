import { expect, it, vi } from "vitest";
import type { UzuRuntime } from "../uzu-runtime";
import { createModelManager } from "./browser-models";
import type { DownloadEvent, ModelsService } from "@/platform/services/models";

function fixture() {
  const free = vi.fn();
  class OnDevice {
    source = new Registry();
    free = free;
  }
  class Registry {
    files = [{ name: "model.safetensors", url: "https://models.example/weights", size: 4096, free }];
    free = free;
  }
  class Failure {
    constructor(public message: string) {}
    free = free;
  }
  let phase = "Not Downloaded";
  let downloaded = 0;
  let failure: string | undefined;
  const state = () => ({
    name: phase,
    downloadedBytes: downloaded,
    totalBytes: 4096,
    phase: failure ? new Failure(failure) : { free },
    free,
  });
  const model = {
    identifier: "vendor:family:small",
    repoIds: ["vendor/model"],
    isDownloadable: true,
    name: "Family Small 4-bit",
    family: {
      identifier: "vendor:family",
      name: "Family",
      free,
      vendor: {
        name: "Vendor",
        free,
        metadata: {
          free,
          icons: [
            { theme: "Light", url: "https://models.example/light.svg", free },
            { theme: "Dark", url: "https://models.example/dark.svg", free },
          ],
        },
      },
    },
    properties: { size: 230000000, free },
    quantization: { method: "mirai-m", bitsPerWeight: 4, free },
    accessibility: new OnDevice(),
    free,
  };
  let callback = () => {};
  let finish: (() => void) | undefined;
  const stream = {
    next: vi.fn(
      () =>
        new Promise<null>((resolve) => {
          finish = () => resolve(null);
        }),
    ),
    free: vi.fn(),
  };
  const downloader = {
    pause: vi.fn(async () => {
      phase = "Paused";
      finish?.();
      callback();
    }),
    delete: vi.fn(async () => {
      phase = "Not Downloaded";
      downloaded = 0;
      finish?.();
      callback();
    }),
    free: vi.fn(),
  };
  const engine = {
    get modelsForChat() {
      return Promise.resolve([model]);
    },
    get downloadStates() {
      return Promise.resolve({ [model.identifier]: state() });
    },
    registerCallback: vi.fn(async (value: { invoke: () => void }) => {
      callback = value.invoke;
    }),
    download: vi.fn(async () => {
      phase = "Downloading";
      callback();
      return stream;
    }),
    downloader: vi.fn(() => downloader),
    modelPath: vi.fn(async () => (phase === "Downloaded" ? "/mirai/models/vendor-family-small" : null)),
    refreshModels: vi.fn(async () => [model]),
  };
  const runtime = {
    engine,
    api: {
      EngineCallback: { create: (invoke: () => void) => ({ invoke, free }) },
      ModelAccessibilityOnDevice: OnDevice,
      ModelSourceRegistry: Registry,
      DownloadPhaseError: Failure,
      ImageTheme: { Light: "Light", Dark: "Dark" },
    },
  } as unknown as UzuRuntime;
  return {
    runtime,
    engine,
    downloader,
    stream,
    model,
    update(name: string, bytes: number, message?: string) {
      phase = name;
      downloaded = bytes;
      failure = message;
      callback();
      if (name === "Downloaded" || name === "Error") finish?.();
    },
  };
}

it("uses Uzu's catalog, metadata and byte counts without fetching a second JS catalog", async () => {
  const source = fixture();
  const manager = createModelManager(async () => source.runtime);
  expect((await manager.getModels()).models[0]).toMatchObject({
    identifier: "vendor:family:small",
    repoId: "vendor/model",
    name: "Family Small 4-bit",
    vendor: "Vendor",
    familyIdentifier: "vendor:family",
    familyName: "Family",
    paramSize: 230000000,
    vendorIcons: { light: "https://models.example/light.svg", dark: "https://models.example/dark.svg" },
    quantization: "mirai-m",
    quantizationBits: 4,
    files: [{ name: "model.safetensors", sizeBytes: 4096 }],
    state: { phase: "NotDownloaded", totalKbytes: 4, downloadedKbytes: 0 },
  });
  expect(source.engine.registerCallback).toHaveBeenCalledOnce();
});

it("starts one native download and pauses through Downloader rather than a JS AbortController", async () => {
  const source = fixture();
  const manager = createModelManager(async () => source.runtime);
  const changed = vi.fn();
  manager.onModelsChanged(changed);
  await manager.startDownload("vendor/model");
  await manager.startDownload("vendor/model");
  expect(source.engine.download).toHaveBeenCalledOnce();
  source.update("Downloading", 2048);
  expect((await manager.getModels()).models[0]?.state).toMatchObject({ phase: "Downloading", downloadedKbytes: 2 });
  await manager.pauseDownload("vendor/model");
  expect(source.downloader.pause).toHaveBeenCalledOnce();
  expect(source.downloader.free).toHaveBeenCalledOnce();
  expect(source.stream.free).toHaveBeenCalledOnce();
  expect((await manager.getModels()).models[0]?.state.phase).toBe("Paused");
  expect(changed).toHaveBeenCalled();
});

it("restores progress from native storage after adapter recreation and resumes the same model", async () => {
  const source = fixture();
  source.update("Paused", 2048);
  const first = createModelManager(async () => source.runtime);
  expect((await first.getModels()).models[0]?.state.downloadedKbytes).toBe(2);
  const reloaded = createModelManager(async () => source.runtime);
  expect((await reloaded.getModels()).models[0]?.state.phase).toBe("Paused");
  await reloaded.startDownload("vendor/model");
  expect(source.engine.download).toHaveBeenCalledWith(source.model);
  source.update("Downloaded", 4096);
  await vi.waitFor(async () => expect((await reloaded.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  expect(await reloaded.modelPath("vendor/model")).toBe("/mirai/models/vendor-family-small");
});

it("deletes through the native downloader and takes the resulting state from Uzu", async () => {
  const source = fixture();
  source.update("Downloaded", 4096);
  const manager = createModelManager(async () => source.runtime);
  await manager.deleteModel("vendor/model");
  expect(source.downloader.delete).toHaveBeenCalledOnce();
  expect((await manager.getModels()).models[0]?.state).toMatchObject({ phase: "NotDownloaded", downloadedKbytes: 0 });
  expect(await manager.modelPath("vendor/model")).toBeNull();
});

it("preserves a native download error and allows a retry", async () => {
  const source = fixture();
  const manager = createModelManager(async () => source.runtime);
  source.engine.download.mockRejectedValueOnce(new Error("Storage is full"));
  await manager.startDownload("vendor/model");
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state).toMatchObject({ phase: "Error", error: "Storage is full" }),
  );
  await manager.startDownload("vendor/model");
  expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloading");
  await manager.pauseDownload("vendor/model");
});

it("maps Rust error variants and retries initialization after a rejected WASM load", async () => {
  const source = fixture();
  source.update("Error", 2048, "Network disconnected");
  const load = vi.fn(async () => source.runtime).mockRejectedValueOnce(new Error("WASM unavailable"));
  const manager = createModelManager(load);
  await expect(manager.getModels()).rejects.toThrow("WASM unavailable");
  expect((await manager.getModels()).models[0]?.state).toMatchObject({
    phase: "Error",
    error: "Network disconnected",
    downloadedKbytes: 2,
  });
  expect(load).toHaveBeenCalledTimes(2);
});

it("refreshes through Uzu and unsubscribes catalog listeners", async () => {
  const source = fixture();
  const manager = createModelManager(async () => source.runtime);
  const changed = vi.fn();
  const unsubscribe = manager.onModelsChanged(changed);
  await manager.refreshModels();
  expect(source.engine.refreshModels).toHaveBeenCalledOnce();
  expect(changed).toHaveBeenCalled();
  unsubscribe();
  changed.mockClear();
  source.update("Paused", 1024);
  expect(changed).not.toHaveBeenCalled();
});

it("does not hide a terminal native update behind an older pending snapshot", async () => {
  const source = fixture();
  const saved = await source.engine.downloadStates;
  let resolveOld!: (value: typeof saved) => void;
  const delayed = new Promise<typeof saved>((resolve) => {
    resolveOld = resolve;
  });
  const states = vi.spyOn(source.engine, "downloadStates", "get").mockReturnValueOnce(delayed);
  const manager = createModelManager(async () => source.runtime);
  const before = manager.getModels();
  await vi.waitFor(() => expect(states).toHaveBeenCalledOnce());
  source.update("Downloaded", 4096);
  const after = await manager.getModels();
  expect(after.models[0]?.state.phase).toBe("Downloaded");
  resolveOld(saved);
  expect((await before).models[0]!.state.seq).toBeLessThan(after.models[0]!.state.seq);
});

it("implements ModelsService events with repo IDs and increasing sequences, including resume and deletion", async () => {
  const source = fixture();
  const manager: ModelsService = createModelManager(async () => source.runtime);
  const events: DownloadEvent[] = [];
  const unsubscribe = manager.onDownloadEvent((event) => events.push(event));
  await manager.getModels();
  await manager.startDownload("vendor/model");
  await vi.waitFor(() => expect(events.at(-1)).toMatchObject({ kind: "state", phase: "Downloading" }));
  source.update("Downloading", 2048);
  await vi.waitFor(() =>
    expect(events.at(-1)).toMatchObject({ kind: "state", completedBytes: 2048, totalBytes: 4096 }),
  );
  await manager.pauseDownload("vendor/model");
  await vi.waitFor(() => expect(events.at(-1)?.kind).toBe("paused"));
  await manager.resumeDownload("vendor/model");
  source.update("Downloaded", 4096);
  await vi.waitFor(() => expect(events.at(-1)?.kind).toBe("done"));
  await manager.deleteModel("vendor/model");
  await vi.waitFor(() => expect(events.at(-1)?.kind).toBe("deleted"));
  expect(events.every((event) => event.identifier === "vendor/model")).toBe(true);
  expect(events.every((event, index) => index === 0 || event.seq > events[index - 1]!.seq)).toBe(true);
  unsubscribe();
  const count = events.length;
  source.update("Paused", 1024);
  await manager.getModels();
  expect(events).toHaveLength(count);
});

it("restores a downloaded model without announcing a new download", async () => {
  const source = fixture();
  source.update("Loading", 0);
  const manager = createModelManager(async () => source.runtime);
  const events: DownloadEvent[] = [];
  manager.onDownloadEvent((event) => events.push(event));
  await manager.getModels();
  source.update("Downloaded", 4096);
  await manager.getModels();
  expect(events.some((event) => event.kind === "state" && event.phase === "Downloaded")).toBe(true);
  expect(events.some((event) => event.kind === "done")).toBe(false);
});
