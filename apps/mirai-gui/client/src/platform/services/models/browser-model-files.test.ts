import "fake-indexeddb/auto";
import { createHash } from "node:crypto";
import { Blob as NodeBlob } from "node:buffer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { browserDb } from "../shared/browser-db";
import {
  deleteModelFile,
  downloadModelFile,
  modelFileStatus,
  openModelFileBlob,
  readModelFile,
  MODEL_CHUNK_BYTES,
} from "./browser-model-files";
import type { BrowserModelFile } from "./browser-catalog";

let cached: Map<string, Response>;
let fetchMock: ReturnType<typeof vi.fn>;
const data = new TextEncoder().encode("123456789");
const file = (method: "sha256" | "git_blob_sha1" | "crc32c" = "sha256"): BrowserModelFile => ({
  name: "weights",
  url: "https://models.example/pinned/weights",
  sizeBytes: data.length,
  hashes: [
    {
      method,
      value:
        method === "crc32c"
          ? "4waSgw=="
          : createHash(method === "sha256" ? "sha256" : "sha1")
              .update(method === "git_blob_sha1" ? `blob ${data.length}\0` : "")
              .update(data)
              .digest("hex"),
    },
  ],
});
const download = (modelFile = file()) => downloadModelFile(modelFile, new AbortController().signal, () => undefined);
beforeEach(async () => {
  vi.stubGlobal("Blob", NodeBlob);
  for (const key of await browserDb.keys("modelStates")) await browserDb.delete("modelStates", key);
  cached = new Map();
  vi.stubGlobal("caches", {
    open: async () => ({
      match: async (url: string) => cached.get(url)?.clone(),
      put: async (url: string, response: Response) => {
        cached.set(url, new Response(await response.arrayBuffer(), { headers: response.headers }));
      },
      delete: async (url: string) => cached.delete(url),
    }),
  });
  fetchMock = vi.fn(async () => new Response(data));
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

for (const method of ["sha256", "git_blob_sha1", "crc32c"] as const) {
  it(`checks ${method} against an independent known digest`, async () => {
    await download(file(method));
    expect((await modelFileStatus(file(method))).complete).toBe(true);
    expect([...(await readModelFile(file(method)))]).toEqual([...data]);
  });
}
it("restarts safely when a server ignores the resume range", async () => {
  fetchMock.mockResolvedValueOnce(new Response(data.slice(0, 3)));
  await expect(download()).rejects.toThrow("Incomplete download");
  fetchMock.mockResolvedValueOnce(new Response(data, { status: 200 }));
  await download();
  expect(fetchMock.mock.calls[1]?.[1].headers).toEqual({ Range: "bytes=3-" });
  expect([...(await readModelFile(file()))]).toEqual([...data]);
  expect(cached.size).toBe(1);
});
it("rejects the wrong Content-Range before caching response bytes", async () => {
  fetchMock.mockResolvedValueOnce(new Response(data.slice(0, 3)));
  await expect(download()).rejects.toThrow("Incomplete download");
  fetchMock.mockResolvedValueOnce(
    new Response(data.slice(3), { status: 206, headers: { "content-range": "bytes 2-8/9" } }),
  );
  await expect(download()).rejects.toThrow("wrong byte range");
  expect((await modelFileStatus(file())).completedBytes).toBe(3);
});
it("rejects changed ETags and clears an outdated prefix", async () => {
  fetchMock.mockResolvedValueOnce(new Response(data.slice(0, 3), { headers: { etag: '"first"' } }));
  await expect(download()).rejects.toThrow("Incomplete download");
  fetchMock.mockResolvedValueOnce(new Response(data.slice(3), { status: 206, headers: { etag: '"changed"' } }));
  await expect(download()).rejects.toThrow("changed on the server");
  expect((await modelFileStatus(file())).completedBytes).toBe(0);
  expect(cached.size).toBe(0);
});
it("splits a large network chunk into bounded durable chunks", async () => {
  const large = new Uint8Array(MODEL_CHUNK_BYTES + 31).fill(0x53);
  const modelFile = {
    ...file(),
    sizeBytes: large.length,
    hashes: [{ method: "sha256" as const, value: createHash("sha256").update(large).digest("hex") }],
  };
  fetchMock.mockResolvedValueOnce(new Response(large));
  await download(modelFile);
  expect(cached.size).toBe(2);
  expect((await modelFileStatus(modelFile)).completedBytes).toBe(large.length);
  expect(
    createHash("sha256")
      .update(await readModelFile(modelFile))
      .digest("hex"),
  ).toBe(modelFile.hashes[0]!.value);
});
it("does not duplicate data when a chunk metadata transaction initially fails", async () => {
  const original = browserDb.put.bind(browserDb);
  let failed = false;
  vi.spyOn(browserDb, "put").mockImplementation(async (store, key, value) => {
    if (!failed && key.startsWith("file:")) {
      failed = true;
      throw new DOMException("Storage full", "QuotaExceededError");
    }
    return original(store, key, value);
  });
  await expect(download()).rejects.toThrow("Storage full");
  await download();
  expect([...(await readModelFile(file()))]).toEqual([...data]);
  expect((await modelFileStatus(file())).completedBytes).toBe(data.length);
});
it("does not report a model file complete after browser cache eviction", async () => {
  await download();
  cached.clear();
  expect((await modelFileStatus(file())).complete).toBe(false);
  await expect(readModelFile(file())).rejects.toThrow("Download weights");
  await expect(openModelFileBlob(file())).rejects.toThrow("Download weights");
});
it("opens weight chunks as a Blob without materializing a whole checkpoint array", async () => {
  const large = Uint8Array.from({ length: MODEL_CHUNK_BYTES + 31 }, (_, index) => index % 251);
  const modelFile = {
    ...file(),
    sizeBytes: large.length,
    hashes: [{ method: "sha256" as const, value: createHash("sha256").update(large).digest("hex") }],
  };
  fetchMock.mockResolvedValueOnce(new Response(large));
  await download(modelFile);
  const arrays = vi.spyOn(Response.prototype, "arrayBuffer").mockRejectedValue(new Error("Unexpected full array"));
  const blob = await openModelFileBlob(modelFile);
  expect(blob.size).toBe(large.length);
  expect(arrays).not.toHaveBeenCalled();
  expect(new Uint8Array(await blob.slice(MODEL_CHUNK_BYTES - 7).arrayBuffer())).toEqual(
    large.slice(MODEL_CHUNK_BYTES - 7),
  );
  // An opened model owns an immutable snapshot even if another tab deletes its cache.
  await deleteModelFile(modelFile);
  expect(cached.size).toBe(0);
  expect(new Uint8Array(await blob.slice(0, 19).arrayBuffer())).toEqual(large.slice(0, 19));
});
it("preserves nonuniform resumed chunks in the composite Blob", async () => {
  fetchMock.mockResolvedValueOnce(new Response(data.slice(0, 3)));
  await expect(download()).rejects.toThrow("Incomplete download");
  await expect(openModelFileBlob(file())).rejects.toThrow("Download weights");
  fetchMock.mockResolvedValueOnce(
    new Response(data.slice(3), { status: 206, headers: { "content-range": "bytes 3-8/9" } }),
  );
  await download();
  const blob = await openModelFileBlob(file());
  expect(blob.size).toBe(9);
  expect(await blob.slice(2, 6).text()).toBe("3456");
});
it("can cancel cached Blob assembly between chunks without damaging the saved download", async () => {
  const bytes = new Uint8Array(MODEL_CHUNK_BYTES + 1).fill(0x53);
  const modelFile = {
    ...file(),
    sizeBytes: bytes.length,
    hashes: [{ method: "sha256" as const, value: createHash("sha256").update(bytes).digest("hex") }],
  };
  fetchMock.mockResolvedValueOnce(new Response(bytes));
  await download(modelFile);
  let cancelled = false;
  const original = Response.prototype.blob;
  const reads = vi.spyOn(Response.prototype, "blob").mockImplementation(async function (this: Response) {
    const value = await original.call(this);
    cancelled = true;
    return value;
  });
  await expect(
    openModelFileBlob(modelFile, () => {
      if (cancelled) throw new Error("Model loading cancelled");
    }),
  ).rejects.toThrow("Model loading cancelled");
  expect(reads).toHaveBeenCalledOnce();
  expect((await modelFileStatus(modelFile)).complete).toBe(true);
  reads.mockRestore();
  expect((await openModelFileBlob(modelFile)).size).toBe(bytes.length);
});
it("preserves file lengths and Blob slice offsets beyond 2 GiB", async () => {
  // Reusing one immutable part keeps this test's allocation at 4 MiB while the
  // resulting Blob has a real length above 2 GiB. Node limits Blob length to
  // 4 GiB; the real browser integration covers larger checkpoints.
  const part = new Blob([new Uint8Array(MODEL_CHUNK_BYTES).fill(0x53)]);
  const chunks = 513;
  const modelFile = { ...file(), sizeBytes: MODEL_CHUNK_BYTES * chunks };
  await browserDb.put("modelStates", `file:${modelFile.url}`, {
    signature: JSON.stringify([modelFile.sizeBytes, modelFile.hashes]),
    chunks,
    completedBytes: modelFile.sizeBytes,
    complete: true,
  });
  vi.stubGlobal("caches", {
    open: async () => ({
      match: async () => ({
        headers: new Headers({ "content-length": String(part.size) }),
        blob: async () => part,
      }),
    }),
  });
  const blob = await openModelFileBlob(modelFile);
  expect(blob.size).toBe(2_151_677_952);
  expect(new Uint8Array(await blob.slice(2 ** 31 - 3, 2 ** 31 + 3).arrayBuffer())).toEqual(
    new Uint8Array(6).fill(0x53),
  );
});
it("rejects cached blob bytes that disagree with the saved chunk length", async () => {
  await download();
  const key = [...cached.keys()][0]!;
  cached.set(key, new Response("bad", { headers: { "content-length": "9" } }));
  await expect(openModelFileBlob(file())).rejects.toThrow("Incomplete cached file");
  expect((await modelFileStatus(file())).complete).toBe(false);
});
it("does not expose a Blob from a different checkpoint revision", async () => {
  await download();
  const changed = { ...file(), hashes: [{ method: "sha256" as const, value: "a".repeat(64) }] };
  await expect(openModelFileBlob(changed)).rejects.toThrow("Download weights");
  expect(cached.size).toBe(0);
});
it("refuses files without a supported integrity checksum", async () => {
  await expect(download({ ...file(), hashes: [] })).rejects.toThrow("no integrity checksum");
  expect(fetchMock).not.toHaveBeenCalled();
});
it("reports repository access errors without saving a complete file", async () => {
  fetchMock.mockResolvedValueOnce(new Response("denied", { status: 403 }));
  await expect(download()).rejects.toThrow("Access to weights was denied");
  expect((await modelFileStatus(file())).complete).toBe(false);
});
