import { createCRC32, createSHA1, createSHA256 } from "hash-wasm";
import { browserDb } from "../shared/browser-db";
import type { BrowserModelFile } from "./browser-catalog";

export const MODEL_CACHE = "mirai.web.models.v1";
export const MODEL_CHUNK_BYTES = 4 * 1024 * 1024;
type FileState = { signature: string; chunks: number; completedBytes: number; complete: boolean; etag?: string };
const stateKey = (file: BrowserModelFile) => `file:${file.url}`;
const signature = (file: BrowserModelFile) => JSON.stringify([file.sizeBytes, file.hashes]);
const chunkKey = (file: BrowserModelFile, chunk: number) => {
  const url = new URL(file.url);
  url.searchParams.set("__mirai_chunk", String(chunk));
  return url.href;
};
const emptyState = (file: BrowserModelFile): FileState => ({
  signature: signature(file),
  chunks: 0,
  completedBytes: 0,
  complete: false,
});

async function savedState(file: BrowserModelFile): Promise<FileState> {
  const state = await browserDb.get<FileState>("modelStates", stateKey(file));
  if (!state) return emptyState(file);
  if (
    state.signature !== signature(file) ||
    !Number.isSafeInteger(state.chunks) ||
    state.chunks < 0 ||
    state.completedBytes > file.sizeBytes
  ) {
    await deleteModelFile(file);
    return emptyState(file);
  }
  return state;
}

export async function modelFileStatus(file: BrowserModelFile, checkCancelled?: () => void): Promise<FileState> {
  checkCancelled?.();
  const state = await savedState(file);
  if (!state.chunks) return state;
  const cache = await caches.open(MODEL_CACHE);
  let bytes = 0;
  for (let chunk = 0; chunk < state.chunks; chunk++) {
    checkCancelled?.();
    const stored = await cache.match(chunkKey(file, chunk));
    const length = Number(stored?.headers.get("content-length"));
    if (!stored || !Number.isSafeInteger(length) || length <= 0 || length > MODEL_CHUNK_BYTES) {
      await deleteModelFile(file);
      return emptyState(file);
    }
    bytes += length;
  }
  if (bytes !== state.completedBytes || (state.complete && bytes !== file.sizeBytes)) {
    await deleteModelFile(file);
    return emptyState(file);
  }
  return state;
}

export async function deleteModelFile(file: BrowserModelFile): Promise<void> {
  const state = await browserDb.get<FileState>("modelStates", stateKey(file));
  const cache = await caches.open(MODEL_CACHE);
  // The extra slot cleans a chunk committed immediately before a tab closed,
  // when its atomic metadata update may not have completed yet.
  for (let chunk = 0; chunk <= (state?.chunks ?? 0); chunk++) await cache.delete(chunkKey(file, chunk));
  await cache.delete(file.url); // Earlier browser builds cached whole responses.
  await browserDb.delete("modelStates", stateKey(file));
}

export async function readModelFile(
  file: BrowserModelFile,
  checkCancelled?: () => void,
): Promise<Uint8Array<ArrayBuffer>> {
  checkCancelled?.();
  const state = await modelFileStatus(file, checkCancelled);
  if (!state.complete) throw new Error(`Download ${file.name} before starting a chat`);
  const cache = await caches.open(MODEL_CACHE);
  const result = new Uint8Array(file.sizeBytes);
  let offset = 0;
  for (let chunk = 0; chunk < state.chunks; chunk++) {
    checkCancelled?.();
    const response = await cache.match(chunkKey(file, chunk));
    if (!response) throw new Error(`The browser removed ${file.name}; download it again`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    result.set(bytes, offset);
    offset += bytes.byteLength;
  }
  if (offset !== file.sizeBytes) throw new Error(`Incomplete cached file: ${file.name}`);
  return result;
}

// Keep downloaded checkpoint bytes in browser-managed Blob storage. Materializing
// one full Uint8Array here would duplicate the entire model before WASM can read it.
export async function openModelFileBlob(file: BrowserModelFile, checkCancelled?: () => void): Promise<Blob> {
  checkCancelled?.();
  const state = await modelFileStatus(file, checkCancelled);
  if (!state.complete) throw new Error(`Download ${file.name} before starting a chat`);
  const cache = await caches.open(MODEL_CACHE);
  const parts: Blob[] = [];
  let size = 0;
  for (let chunk = 0; chunk < state.chunks; chunk++) {
    checkCancelled?.();
    const response = await cache.match(chunkKey(file, chunk));
    if (!response) throw new Error(`The browser removed ${file.name}; download it again`);
    const blob = await response.blob();
    if (blob.size !== Number(response.headers.get("content-length"))) {
      await deleteModelFile(file);
      throw new Error(`Incomplete cached file: ${file.name}; download it again`);
    }
    parts.push(blob);
    size += blob.size;
  }
  if (size !== file.sizeBytes) throw new Error(`Incomplete cached file: ${file.name}`);
  return new Blob(parts, { type: "application/octet-stream" });
}

async function verifier(file: BrowserModelFile) {
  const expected =
    file.hashes.find((hash) => hash.method === "sha256") ??
    file.hashes.find((hash) => hash.method === "git_blob_sha1") ??
    file.hashes.find((hash) => hash.method === "crc32c");
  if (!expected) throw new Error(`The catalog provides no integrity checksum for ${file.name}`);
  const hash =
    expected.method === "sha256"
      ? await createSHA256()
      : expected.method === "git_blob_sha1"
        ? await createSHA1()
        : await createCRC32(0x82f63b78);
  hash.init();
  if (expected.method === "git_blob_sha1") hash.update(new TextEncoder().encode(`blob ${file.sizeBytes}\0`));
  return {
    update: (bytes: Uint8Array) => hash.update(bytes),
    matches: () => {
      if (expected.method !== "crc32c") return hash.digest("hex") === expected.value;
      const binary = hash.digest("binary");
      return btoa(String.fromCharCode(...binary)) === expected.value;
    },
  };
}

export async function downloadModelFile(
  file: BrowserModelFile,
  signal: AbortSignal,
  onProgress: (completedBytes: number) => void,
): Promise<void> {
  let state = await modelFileStatus(file);
  if (state.complete) {
    onProgress(file.sizeBytes);
    return;
  }
  const cache = await caches.open(MODEL_CACHE);
  let hash = await verifier(file);
  for (let chunk = 0; chunk < state.chunks; chunk++) {
    signal.throwIfAborted();
    const response = await cache.match(chunkKey(file, chunk));
    if (!response) throw new Error(`Incomplete cached file: ${file.name}`);
    hash.update(new Uint8Array(await response.arrayBuffer()));
  }
  onProgress(state.completedBytes);
  let pending = new Uint8Array(MODEL_CHUNK_BYTES);
  let pendingBytes = 0;
  const commit = async () => {
    if (!pendingBytes) return;
    const bytes = pending.slice(0, pendingBytes);
    await cache.put(
      chunkKey(file, state.chunks),
      new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } }),
    );
    const next = { ...state, chunks: state.chunks + 1, completedBytes: state.completedBytes + bytes.byteLength };
    await browserDb.put("modelStates", stateKey(file), next);
    hash.update(bytes);
    state = next;
    pendingBytes = 0;
    pending = new Uint8Array(MODEL_CHUNK_BYTES);
  };
  if (state.completedBytes < file.sizeBytes) {
    signal.throwIfAborted();
    const headers: Record<string, string> = {};
    if (state.completedBytes > 0) {
      headers.Range = `bytes=${state.completedBytes}-`;
      // Pinned URLs and final hashes protect servers that do not expose ETag.
      // Avoid If-Range: it would require a preflight on otherwise simple CORS GETs.
    }
    const response = await fetch(file.url, { signal, headers });
    if (!response.ok || !response.body) {
      if (response.status === 401 || response.status === 403)
        throw new Error(`Access to ${file.name} was denied. This model may require repository access.`);
      throw new Error(`Could not download ${file.name} (HTTP ${response.status})`);
    }
    if (state.completedBytes > 0 && response.status === 200) {
      // A server may ignore Range; discard the prefix before consuming its full response.
      await deleteModelFile(file);
      state = emptyState(file);
      hash = await verifier(file);
      onProgress(0);
    }
    const range = response.headers.get("content-range");
    if (response.status === 206 && range) {
      const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(range);
      if (!match || Number(match[1]) !== state.completedBytes || Number(match[3]) !== file.sizeBytes) {
        await response.body.cancel();
        throw new Error(`The server returned the wrong byte range for ${file.name}`);
      }
    }
    const etag = response.headers.get("etag");
    if (state.completedBytes > 0 && state.etag && etag && state.etag !== etag) {
      await response.body.cancel();
      await deleteModelFile(file);
      throw new Error(`${file.name} changed on the server. Retry its download.`);
    }
    if (etag && !etag.startsWith("W/")) state.etag = etag;
    const reader = response.body.getReader();
    const abort = () => {
      void reader.cancel(signal.reason).catch(() => undefined);
    };
    signal.addEventListener("abort", abort, { once: true });
    let readError: unknown;
    try {
      while (true) {
        signal.throwIfAborted();
        const { value, done } = await reader.read();
        if (done) break;
        if (state.completedBytes + pendingBytes + value.byteLength > file.sizeBytes)
          throw new Error(`The server returned too many bytes for ${file.name}`);
        let offset = 0;
        while (offset < value.byteLength) {
          const count = Math.min(value.byteLength - offset, MODEL_CHUNK_BYTES - pendingBytes);
          pending.set(value.subarray(offset, offset + count), pendingBytes);
          pendingBytes += count;
          offset += count;
          if (pendingBytes === MODEL_CHUNK_BYTES) await commit();
        }
        onProgress(state.completedBytes + pendingBytes);
      }
    } catch (error) {
      readError = error;
    } finally {
      // Keep even the final partial chunk on pause or a broken connection.
      signal.removeEventListener("abort", abort);
      await reader.cancel().catch(() => undefined);
      await commit();
    }
    if (readError) throw readError;
  }
  signal.throwIfAborted();
  if (state.completedBytes !== file.sizeBytes) throw new Error(`Incomplete download of ${file.name}`);
  if (!hash.matches()) {
    await deleteModelFile(file);
    throw new Error(`Integrity check failed for ${file.name}. Retry the download.`);
  }
  await browserDb.put("modelStates", stateKey(file), { ...state, complete: true });
  onProgress(file.sizeBytes);
}
