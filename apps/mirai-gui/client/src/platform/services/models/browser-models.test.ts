import "fake-indexeddb/auto";
import { createHash } from "node:crypto";
import { Blob as NodeBlob } from "node:buffer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { browserDb } from "../shared/browser-db";
import type { DownloadEvent } from ".";

const bytes: Record<string, string> = { "config.json": "{}", "tokenizer.json": "tok", "model.safetensors": "weights" };
const files = Object.entries(bytes).map(([name, body]) => ({
  name,
  url: `https://models.example/${name}`,
  size: body.length,
  hashes: [{ method: "sha256", value: createHash("sha256").update(body).digest("hex") }],
}));
const identifier = "vendor:family:tiny";
const repoId = "vendor/test-model";
const registry = {
  metadatas: [
    { id: "vendor", name: "Test", icons: [] },
    { id: "family", name: "Test family", icons: [] },
    { id: "size", name: "Tiny", icons: [] },
  ],
  models: [
    {
      id: identifier,
      family: { id: "test", metadata_id: "family", vendor: { metadata_id: "vendor" } },
      properties: { metadata_id: "size", size: 230000000 },
      specializations: [{ type: "chat", supports_reasoning: false }],
      accessibility: { type: "local", reference: { type: "mirai", repository: { identifier: repoId }, files } },
    },
  ],
};
let cached: Map<string, Response>;
let fetchMock: ReturnType<typeof vi.fn>;
const originalLocks = navigator.locks;
const originalStorage = navigator.storage;

beforeEach(async () => {
  vi.stubGlobal("Blob", NodeBlob);
  vi.resetModules();
  for (const key of await browserDb.keys("modelStates")) await browserDb.delete("modelStates", key);
  cached = new Map();
  vi.stubGlobal("caches", {
    open: async () => ({
      match: async (url: string) => cached.get(url)?.clone(),
      put: async (url: string, response: Response) => {
        const body = await response.arrayBuffer();
        cached.set(url, new Response(body, { headers: response.headers }));
      },
      delete: async (url: string) => cached.delete(url),
    }),
  });
  Object.defineProperty(navigator, "gpu", { value: {}, configurable: true });
  Object.defineProperty(navigator, "locks", { value: undefined, configurable: true });
  fetchMock = vi.fn(async (url: string) => {
    if (url.endsWith("fetch/models")) return Response.json(registry);
    return new Response(bytes[url.split("/").at(-1)!]);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  Object.defineProperty(navigator, "locks", { configurable: true, value: originalLocks });
  Object.defineProperty(navigator, "storage", { configurable: true, value: originalStorage });
});

it("uses the live desktop registry protocol and verifies cached files for offline generation", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  const events: DownloadEvent[] = [];
  manager.onDownloadEvent((event) => events.push(event));
  expect((await manager.getModels()).models[0]?.state.phase).toBe("NotDownloaded");
  expect(fetchMock.mock.calls[0]?.[0]).toBe("https://sdk.trymirai.com/api/v1/fetch/models");
  expect(JSON.parse(fetchMock.mock.calls[0]?.[1].body)).toMatchObject({
    backends: [{ identifier: "uzu", version: "0.17.0" }],
    device: { os_name: "browser", cpu_name: null },
  });
  await manager.startDownload(repoId);
  await vi.waitFor(() => expect(events.at(-1)).toMatchObject({ kind: "done", identifier: repoId }));
  expect(events.every((event) => event.identifier === repoId)).toBe(true);
  expect((await manager.getModels()).models[0]?.state.downloadedKbytes).toBe(12 / 1024);
  vi.resetModules();
  fetchMock.mockRejectedValue(new TypeError("Offline"));
  const reloaded = await import("./browser-models");
  expect((await reloaded.browserModelManager.getModels()).models[0]?.state.phase).toBe("Downloaded");
  const stored = await reloaded.cachedModelFiles(await reloaded.findBrowserModel(repoId));
  expect(stored["model.safetensors"]).toBeInstanceOf(Blob);
  expect(await stored["model.safetensors"].text()).toBe("weights");
  expect(new TextDecoder().decode(stored["config.json"])).toBe("{}");
  expect(new TextDecoder().decode(stored["tokenizer.json"])).toBe("tok");
  await vi.waitFor(async () => expect((await reloaded.browserModelManager.getModels()).complete).toBe(false));
});

it("keeps partial weight bytes across pause and reload, then resumes with a byte range", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  let pending = false;
  fetchMock.mockImplementation(async (url: string) => {
    if (url.endsWith("fetch/models")) return Response.json(registry);
    if (!url.endsWith("model.safetensors")) return new Response(bytes[url.split("/").at(-1)!]);
    pending = true;
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("wei"));
        },
      }),
      { headers: { etag: '"revision1"' } },
    );
  });
  await manager.startDownload(repoId);
  await vi.waitFor(() => expect(pending).toBe(true));
  // Let the body chunk reach the download reader before pausing.
  await vi.waitFor(() => expect(fetchMock.mock.calls.length).toBe(4));
  await manager.pauseDownload(repoId);
  expect((await manager.getModels()).models[0]?.state).toMatchObject({ phase: "Paused", downloadedKbytes: 8 / 1024 });
  vi.resetModules();
  const resumed = (await import("./browser-models")).browserModelManager;
  fetchMock.mockImplementation(async (url: string, options?: RequestInit) => {
    if (url.endsWith("fetch/models")) return Response.json(registry);
    expect(options?.headers).toMatchObject({ Range: "bytes=3-" });
    return new Response("ghts", { status: 206, headers: { "content-range": "bytes 3-6/7", etag: '"revision1"' } });
  });
  await resumed.startDownload(repoId);
  await vi.waitFor(async () => expect((await resumed.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("config.json"))).toHaveLength(1);
  expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("model.safetensors"))).toHaveLength(2);
});

it("rejects checksum mismatches and removes invalid bytes before retrying", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  fetchMock.mockImplementation(async (url: string) =>
    url.endsWith("fetch/models")
      ? Response.json(registry)
      : new Response(url.endsWith("model.safetensors") ? "corrupt" : bytes[url.split("/").at(-1)!]),
  );
  await manager.startDownload(repoId);
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state).toMatchObject({
      phase: "Error",
      error: "Integrity check failed for model.safetensors. Retry the download.",
    }),
  );
  expect([...cached.keys()].some((url) => url.includes("model.safetensors"))).toBe(false);
  await manager.deleteModel(repoId);
  expect((await manager.getModels()).models[0]?.state.phase).toBe("NotDownloaded");
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(0);
});

it("retains truncated but valid prefixes for a subsequent retry", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  fetchMock.mockImplementation(async (url: string) =>
    url.endsWith("fetch/models")
      ? Response.json(registry)
      : new Response(url.endsWith("model.safetensors") ? "wei" : bytes[url.split("/").at(-1)!]),
  );
  await manager.startDownload(repoId);
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state).toMatchObject({
      phase: "Error",
      error: "Incomplete download of model.safetensors",
      downloadedKbytes: 8 / 1024,
    }),
  );
  fetchMock.mockImplementation(async (_url: string, options?: RequestInit) => {
    expect(options?.headers).toMatchObject({ Range: "bytes=3-" });
    return new Response("ghts", { status: 206 });
  });
  await manager.startDownload(repoId);
  await vi.waitFor(async () => expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded"));
});

it("trusts the reported quota only below Chrome's 10 GiB headroom cap", async () => {
  const { fitsReportedQuota } = await import("./browser-models");
  const GiB = 1024 ** 3;
  const estimate = (headroom: number) => ({ usage: 3 * GiB, quota: 3 * GiB + headroom });
  expect(fitsReportedQuota(15 * GiB, estimate(10 * GiB))).toBe(true);
  expect(fitsReportedQuota(27 * GiB, estimate(19 * GiB))).toBe(true);
  expect(fitsReportedQuota(15 * GiB, estimate(8 * GiB))).toBe(false);
  expect(fitsReportedQuota(5 * GiB, estimate(8 * GiB))).toBe(true);
  expect(fitsReportedQuota(2 * GiB, estimate(1 * GiB))).toBe(false);
  expect(fitsReportedQuota(15 * GiB, undefined)).toBe(true);
});

it("rejects a download that does not fit a real reported quota before any file request", async () => {
  Object.defineProperty(navigator, "storage", {
    configurable: true,
    value: { estimate: async () => ({ quota: 10, usage: 5 }), persist: async () => false },
  });
  const { browserModelManager: manager } = await import("./browser-models");
  await manager.startDownload(repoId);
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state.error).toContain("Not enough browser storage"),
  );
  expect(fetchMock.mock.calls.filter(([url]) => String(url).startsWith("https://models.example/"))).toHaveLength(0);
});

it("requests persistent storage before checking the quota and downloading", async () => {
  const calls: string[] = [];
  Object.defineProperty(navigator, "storage", {
    configurable: true,
    value: {
      persist: async () => (calls.push("persist"), true),
      estimate: async () => (calls.push("estimate"), { quota: 10 * 1024 ** 3, usage: 0 }),
    },
  });
  const { browserModelManager: manager } = await import("./browser-models");
  await manager.startDownload(repoId);
  await vi.waitFor(async () => expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  expect(calls.slice(0, 2)).toEqual(["persist", "estimate"]);
});

it("reports a storage failure when the chunk cache rejects a write with Chrome's UnknownError", async () => {
  const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const failure = new DOMException("Failed to execute 'put' on 'Cache': Unexpected internal error.", "UnknownError");
  const put = vi.fn(async (url: string, response: Response) => {
    if (url.includes("model.safetensors")) throw failure;
    cached.set(url, new Response(await response.arrayBuffer(), { headers: response.headers }));
  });
  vi.stubGlobal("caches", {
    open: async () => ({
      match: async (url: string) => cached.get(url)?.clone(),
      put,
      delete: async (url: string) => cached.delete(url),
    }),
  });
  const { browserModelManager: manager } = await import("./browser-models");
  await manager.startDownload(repoId);
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state).toMatchObject({
      phase: "Error",
      error: expect.stringContaining("could not save model data"),
      downloadedKbytes: (bytes["config.json"]!.length + bytes["tokenizer.json"]!.length) / 1024,
    }),
  );
  await vi.waitFor(() =>
    expect(log).toHaveBeenCalledWith("[downloads] download failed", expect.objectContaining({ identifier }), failure),
  );
});

it("shows unsupported checkpoints without offering a download", async () => {
  const large = structuredClone(registry);
  large.models[0]!.accessibility.reference.files[0]!.size = 16 * 1024 * 1024 + 1;
  fetchMock.mockResolvedValue(Response.json(large));
  const { browserModelManager: manager } = await import("./browser-models");
  expect((await manager.getModels()).models[0]?.browserUnsupportedReason).toContain("configuration");
  await expect(manager.startDownload(repoId)).rejects.toThrow("configuration");
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it("validates the engine configuration before downloading tokenizer or weights", async () => {
  const { browserModelManager: manager, setBrowserModelConfigInspector } = await import("./browser-models");
  const inspect = vi.fn(async (config: Uint8Array) => {
    expect(new TextDecoder().decode(config)).toBe("{}");
    throw new Error("This model architecture is not supported by WebGPU");
  });
  setBrowserModelConfigInspector(inspect);
  await manager.startDownload(repoId);
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state).toMatchObject({
      phase: "Error",
      error: "This model architecture is not supported by WebGPU",
    }),
  );
  expect(inspect).toHaveBeenCalledOnce();
  expect(fetchMock.mock.calls.filter(([url]) => String(url).startsWith("https://models.example/"))).toEqual([
    [files[0]!.url, expect.anything()],
  ]);
});

it("coordinates competing tabs so deletion aborts the owning tab before removing partial files", async () => {
  const held = new Map<string, Promise<unknown>>();
  const locks = {
    query: async () => ({ held: [...held.keys()].map((name) => ({ name })) }),
    async request(
      name: string,
      optionsOrRun: object | ((lock: object) => unknown),
      callback?: (lock: object | null) => unknown,
    ) {
      const run = callback ?? (optionsOrRun as (lock: object) => unknown);
      if (held.has(name) && typeof optionsOrRun === "object" && "ifAvailable" in optionsOrRun) return run(null!);
      const previous = held.get(name) ?? Promise.resolve();
      const work = previous.then(() => run({ name }));
      held.set(name, work);
      try {
        return await work;
      } finally {
        if (held.get(name) === work) held.delete(name);
      }
    },
  };
  Object.defineProperty(navigator, "locks", { configurable: true, value: locks });
  class TabChannel {
    static channels: TabChannel[] = [];
    onmessage?: (event: MessageEvent) => void;
    constructor() {
      TabChannel.channels.push(this);
    }
    postMessage(data: unknown) {
      for (const other of TabChannel.channels)
        if (other !== this) queueMicrotask(() => other.onmessage?.({ data } as MessageEvent));
    }
  }
  vi.stubGlobal("BroadcastChannel", TabChannel);
  const cancelStream = vi.fn();
  fetchMock.mockImplementation(async (url: string) => {
    if (url.endsWith("fetch/models")) return Response.json(registry);
    if (!url.endsWith("model.safetensors")) return new Response(bytes[url.split("/").at(-1)!]);
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("wei"));
        },
        cancel: cancelStream,
      }),
    );
  });
  const first = (await import("./browser-models")).browserModelManager;
  await first.getModels();
  await first.startDownload(repoId);
  await vi.waitFor(() =>
    expect(fetchMock.mock.calls.filter(([url]) => url.endsWith("model.safetensors"))).toHaveLength(1),
  );
  vi.resetModules();
  const second = (await import("./browser-models")).browserModelManager;
  await second.getModels();
  const events: DownloadEvent[] = [];
  second.onDownloadEvent((event) => events.push(event));
  await second.startDownload(repoId);
  await vi.waitFor(() => expect(events.some((event) => event.kind === "locked")).toBe(true));
  await second.deleteModel(repoId);
  expect(cancelStream).toHaveBeenCalledOnce();
  await vi.waitFor(async () => expect((await first.getModels()).models[0]?.state.phase).toBe("NotDownloaded"));
  expect((await second.getModels()).models[0]?.state.phase).toBe("NotDownloaded");
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(0);
});

it("restores a remote owner's interrupted download after its tab disappears", async () => {
  const first = (await import("./browser-models")).browserModelManager;
  await first.getModels();
  const model = registry.models[0]!;
  await browserDb.put("modelStates", identifier, {
    phase: "Downloading",
    completedBytes: 0,
    totalBytes: 12,
    seq: 1,
    revision: JSON.stringify(model.accessibility.reference.files.map((file) => [file.url, file.size, file.hashes])),
  });
  let held = true;
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: { query: async () => ({ held: held ? [{ name: `mirai.model.${identifier}` }] : [] }) },
  });
  vi.resetModules();
  const observer = (await import("./browser-models")).browserModelManager;
  expect((await observer.getModels()).models[0]?.state.phase).toBe("Downloading");
  held = false;
  expect((await observer.getModels()).models[0]?.state.phase).toBe("Paused");
});

it.each(["Error", "Paused", "NotDownloaded"] as const)(
  "preserves a remote terminal %s state while the writer is releasing its lock",
  async (phase) => {
    const model = registry.models[0]!;
    await browserDb.put("modelStates", identifier, {
      phase,
      completedBytes: 0,
      totalBytes: 12,
      seq: 1,
      revision: JSON.stringify(model.accessibility.reference.files.map((file) => [file.url, file.size, file.hashes])),
      ...(phase === "Error" ? { error: "Unsupported model configuration" } : {}),
    });
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { query: async () => ({ held: [{ name: `mirai.model.${identifier}` }] }) },
    });
    const manager = (await import("./browser-models")).browserModelManager;
    const state = (await manager.getModels()).models[0]!.state;
    expect(state.phase).toBe(phase);
    if (phase === "Error") expect(state.error).toBe("Unsupported model configuration");
  },
);

it("keeps the failure visible if storage cannot persist the terminal download state", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  const db = (await import("../shared/browser-db")).browserDb;
  const put = db.put.bind(db);
  vi.spyOn(db, "put").mockImplementation(async (store, key, value) => {
    if (key === identifier && (value as { phase?: string }).phase === "Error")
      throw new DOMException("Storage full", "QuotaExceededError");
    return put(store, key, value);
  });
  fetchMock.mockImplementation(async (url: string) => {
    if (url.endsWith("fetch/models")) return Response.json(registry);
    throw new TypeError("Connection lost");
  });
  await manager.startDownload(repoId);
  await vi.waitFor(async () =>
    expect((await manager.getModels()).models[0]?.state).toMatchObject({
      phase: "Error",
      error: expect.stringContaining("run out of storage"),
    }),
  );
});

it("keeps a downloaded checkpoint pinned across catalog updates until it is deleted", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  await manager.startDownload(repoId);
  await vi.waitFor(async () => expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  const updated = structuredClone(registry);
  updated.models[0]!.accessibility.reference.files.forEach((file) => {
    file.url = file.url.replace("models.example/", "models.example/revision2/");
  });
  fetchMock.mockImplementation(async (url: string) =>
    url.endsWith("fetch/models") ? Response.json(updated) : new Response(bytes[url.split("/").at(-1)!]),
  );
  await manager.refreshModels();
  expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded");
  await expect((await import("./browser-models")).findBrowserModel(repoId)).resolves.toMatchObject({
    files: [{ url: files[0]!.url }, { url: files[1]!.url }, { url: files[2]!.url }],
  });
  await manager.deleteModel(repoId);
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(0);
  await manager.refreshModels();
  await manager.startDownload(repoId);
  await vi.waitFor(async () => expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(3);
  expect(
    [...cached.keys()].filter((url) => url.includes("__mirai_chunk")).every((url) => url.includes("revision2")),
  ).toBe(true);
  await manager.deleteModel(repoId);
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(0);
});

it("retains installed models when the refreshed registry no longer lists them", async () => {
  const { browserModelManager: manager, cachedModelFiles, findBrowserModel } = await import("./browser-models");
  await manager.startDownload(repoId);
  await vi.waitFor(async () => expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  fetchMock.mockImplementation(async () => Response.json({ metadatas: [], models: [] }));
  await manager.refreshModels();
  expect((await manager.getModels()).models).toHaveLength(1);
  expect(await (await cachedModelFiles(await findBrowserModel(repoId)))["model.safetensors"].text()).toBe("weights");
  await manager.deleteModel(repoId);
  await manager.refreshModels();
  expect((await manager.getModels()).models).toHaveLength(0);
});

it("reuses an older cached base checkpoint without requiring unused speculator files", async () => {
  const { browserModelManager: manager } = await import("./browser-models");
  await manager.startDownload(repoId);
  await vi.waitFor(async () => expect((await manager.getModels()).models[0]?.state.phase).toBe("Downloaded"));
  const { CATALOG_CACHE_KEY } = await import("./browser-catalog");
  const descriptors = await cached.get(CATALOG_CACHE_KEY)!.clone().json();
  const ancillary = {
    name: "speculator/model.safetensors",
    url: "https://models.example/speculator/model.safetensors",
    sizeBytes: 5,
    hashes: [{ method: "sha256" as const, value: createHash("sha256").update("draft").digest("hex") }],
  };
  fetchMock.mockResolvedValueOnce(new Response("draft"));
  await (await import("./browser-model-files")).downloadModelFile(ancillary, new AbortController().signal, () => {});
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(4);
  // Older cache metadata need not have a download-history entry. Migration
  // must retain deletion coverage before omitting ancillary files from totals.
  await browserDb.delete("modelStates", `model-files:${identifier}`);
  descriptors[0].files.push(ancillary);
  cached.set(CATALOG_CACHE_KEY, Response.json(descriptors));
  vi.resetModules();
  fetchMock.mockRejectedValue(new TypeError("Offline"));
  const reloaded = await import("./browser-models");
  const model = (await reloaded.browserModelManager.getModels()).models[0]!;
  expect(model.state).toMatchObject({ phase: "Downloaded", totalKbytes: 12 / 1024 });
  expect(
    await (await reloaded.cachedModelFiles(await reloaded.findBrowserModel(repoId)))["model.safetensors"].text(),
  ).toBe("weights");
  await reloaded.browserModelManager.deleteModel(repoId);
  expect([...cached.keys()].filter((url) => url.includes("__mirai_chunk"))).toHaveLength(0);
});

it("does not persist transient engine-loading failures as unsupported model metadata", async () => {
  const encoded = {
    ...registry,
    models: registry.models.map((model) => ({ ...model, encodings: [{ type: "hanashi" }] })),
  };
  fetchMock.mockImplementation(async () => Response.json(encoded));
  const first = await import("./browser-models");
  first.setBrowserEncodingInspector(async () => {
    throw new Error("Engine module unavailable");
  });
  const snapshot = await first.browserModelManager.getModels();
  expect(snapshot.models[0]?.browserUnsupportedReason).toContain("Engine module unavailable");
  expect(snapshot.error).toContain("Could not check chat capabilities");
  vi.resetModules();
  fetchMock.mockRejectedValue(new TypeError("Offline"));
  const reloaded = await import("./browser-models");
  expect((await reloaded.browserModelManager.getModels()).models[0]?.browserUnsupportedReason).toBeUndefined();
});

it("reuses verified capabilities when the engine temporarily cannot inspect the unchanged encoding", async () => {
  const encoded = {
    ...registry,
    models: registry.models.map((model) => ({
      ...model,
      specializations: [{ type: "chat", supports_reasoning: true, supports_tools: true }],
      encodings: [{ type: "hanashi" }],
    })),
  };
  fetchMock.mockImplementation(async () => Response.json(encoded));
  const { browserModelManager: manager, setBrowserEncodingInspector } = await import("./browser-models");
  setBrowserEncodingInspector(async () => ({
    reasoning: { kind: "toggle", defaultEffort: "default" },
    supportsTools: true,
  }));
  expect((await manager.getModels()).models[0]?.reasoning.kind).toBe("toggle");
  setBrowserEncodingInspector(async () => {
    throw new Error("Worker stopped");
  });
  await manager.refreshModels();
  expect((await manager.getModels()).models[0]).toMatchObject({
    reasoning: { kind: "toggle" },
    supportsTools: true,
    browserUnsupportedReason: undefined,
  });
});

it("retains checkpoint restrictions after encoding inspection, offline reload, and inspector failure", async () => {
  const encoded = {
    ...registry,
    models: registry.models.map((model) => ({
      ...model,
      specializations: [{ type: "chat", supports_reasoning: true, supports_tools: true }],
      encodings: [{ type: "hanashi", name: "lfm2.5-thinking" }],
    })),
  };
  fetchMock.mockImplementation(async () => Response.json(encoded));
  const first = await import("./browser-models");
  first.setBrowserEncodingInspector(async () => ({ reasoning: { kind: "alwaysOn" }, supportsTools: true }));
  expect((await first.browserModelManager.getModels()).models[0]?.reasoning.kind).toBe("alwaysOn");
  encoded.models[0]!.specializations[0]!.supports_reasoning = false;
  encoded.models[0]!.specializations[0]!.supports_tools = false;
  // Even a cached successful inspection cannot override updated checkpoint flags.
  first.setBrowserEncodingInspector(async () => {
    throw new Error("Worker stopped");
  });
  await first.browserModelManager.refreshModels();
  const restricted = {
    reasoning: { kind: "unsupported" },
    supportsTools: false,
    checkpointCapabilities: { reasoning: false, tools: false },
  };
  expect((await first.browserModelManager.getModels()).models[0]).toMatchObject(restricted);
  expect((await first.browserModelManager.getModels()).models[0]?.browserUnsupportedReason).toBeUndefined();
  first.setBrowserEncodingInspector(async () => ({ reasoning: { kind: "alwaysOn" }, supportsTools: true }));
  await first.browserModelManager.refreshModels();
  expect((await first.browserModelManager.getModels()).models[0]).toMatchObject(restricted);
  vi.resetModules();
  fetchMock.mockRejectedValue(new TypeError("Offline"));
  const reloaded = await import("./browser-models");
  expect((await reloaded.browserModelManager.getModels()).models[0]).toMatchObject(restricted);
  expect((await reloaded.browserModelManager.getModels()).models[0]?.browserUnsupportedReason).toBeUndefined();
});
