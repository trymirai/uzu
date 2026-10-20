import type { EngineModel, ModelDownloadPhase } from "@/types/model-manager";
import type { DownloadEvent } from ".";
import { browserDb } from "../shared/browser-db";
import {
  CATALOG_CACHE_KEY,
  browserInferenceFiles,
  browserModelUnsupportedReason,
  fetchRegistry,
  resolveRepository,
  restrictCheckpointCapabilities,
  type BrowserModel,
} from "./browser-catalog";
import {
  MODEL_CACHE,
  deleteModelFile,
  downloadModelFile,
  modelFileStatus,
  openModelFileBlob,
  readModelFile,
} from "./browser-model-files";
export { MODEL_CACHE } from "./browser-model-files";
export type { BrowserModel, BrowserModelFile } from "./browser-catalog";

type DownloadState = {
  phase: ModelDownloadPhase;
  completedBytes: number;
  totalBytes: number;
  error?: string;
  seq: number;
  revision: string;
};
type ModelMessage =
  { type: "cancel"; identifier: string } | { type: "event"; identifier: string; event: DownloadEvent };
const listeners = new Set<(event: DownloadEvent) => void>();
const changedListeners = new Set<() => void>();
const downloads = new Map<string, { controller: AbortController; done: Promise<void> }>();
const states = new Map<string, DownloadState>();
let catalog: BrowserModel[] | undefined;
let initialCatalog: Promise<BrowserModel[]> | undefined;
let catalogRequest: Promise<void> | undefined;
let catalogComplete = false;
let catalogError: string | undefined;
let inspectEncoding: ((encoding: unknown) => Promise<Pick<BrowserModel, "reasoning" | "supportsTools">>) | undefined;
let inspectConfig: ((config: Uint8Array<ArrayBuffer>) => Promise<void>) | undefined;
export function setBrowserEncodingInspector(inspector: NonNullable<typeof inspectEncoding>) {
  inspectEncoding = inspector;
}
export function setBrowserModelConfigInspector(inspector: NonNullable<typeof inspectConfig>) {
  inspectConfig = inspector;
}
let sequence = 0;
let channel: BroadcastChannel | undefined;
const revision = (model: BrowserModel) =>
  JSON.stringify(model.files.map((file) => [file.url, file.sizeBytes, file.hashes]));
const nextSequence = () => (sequence = Math.max(sequence + 1, Date.now() * 1000));
const lockName = (identifier: string) => `mirai.model.${identifier}`;
const changed = () => changedListeners.forEach((listener) => listener());
const capabilityErrorPrefix = "Could not check this model's chat capabilities:";
const errorText = (error: unknown): string => {
  if (error instanceof DOMException && error.name === "QuotaExceededError")
    return "The browser has run out of storage. Delete a model or free device space, then resume the download.";
  // Chrome reports a full disk this way, but also other storage failures.
  if (error instanceof DOMException && error.name === "UnknownError")
    return "The browser could not save model data. If the device is low on storage, free space, then try again.";
  return error instanceof Error ? error.message : String(error);
};

async function rememberModelFiles(model: BrowserModel): Promise<void> {
  const historyKey = `model-files:${model.identifier}`;
  await browserDb.update<BrowserModel["files"]>("modelStates", historyKey, (previousFiles = []) => [
    ...new Map([...previousFiles, ...model.files].map((file) => [file.url, file])).values(),
  ]);
}

function modelChannel(): BroadcastChannel | undefined {
  if (!channel && typeof window !== "undefined" && "BroadcastChannel" in window) {
    channel = new BroadcastChannel("mirai.web.models");
    channel.onmessage = (message: MessageEvent<ModelMessage>) => {
      const data = message.data;
      if (data.type === "cancel") downloads.get(data.identifier)?.controller.abort();
      else if (data.type === "event") {
        sequence = Math.max(sequence, data.event.seq);
        listeners.forEach((listener) => listener(data.event));
        if (data.event.kind !== "progress") {
          states.delete(data.identifier);
          changed();
        }
      }
    };
  }
  return channel;
}
function emit(model: BrowserModel, event: DownloadEvent): void {
  // Desktop events use repo IDs, matching the client store and notification links.
  const normalized = { ...event, identifier: model.repoId ?? model.identifier };
  listeners.forEach((listener) => listener(normalized));
  modelChannel()?.postMessage({ type: "event", identifier: model.identifier, event: normalized });
}

async function refreshCatalog(): Promise<void> {
  if (catalogRequest) return catalogRequest;
  catalogRequest = (async () => {
    catalogError = undefined;
    changed();
    try {
      const models = await fetchRegistry();
      const previous = catalog;
      let cursor = 0;
      let complete = true;
      await Promise.all(
        Array.from({ length: Math.min(8, models.length) }, async () => {
          while (cursor < models.length) {
            const index = cursor++;
            const model = models[index]!;
            try {
              models[index] = await resolveRepository(
                model,
                previous?.find((entry) => entry.identifier === model.identifier),
              );
            } catch (error) {
              // Registry-provided asset URLs remain a valid source with their own
              // hashes when Hugging Face is temporarily unavailable.
              if (!model.files.length) {
                const cached = previous?.find((entry) => entry.identifier === model.identifier);
                if (cached) models[index] = cached;
                complete = false;
                catalogError = errorText(error);
              }
            }
            const cached = previous?.find((entry) => entry.identifier === model.identifier);
            // A catalog update must not replace a usable pinned checkpoint or
            // invalidate a partial download. Deleting it allows the next revision.
            if (
              cached &&
              revision(cached) !== revision(models[index]!) &&
              (downloads.has(cached.identifier) || (await storedProgress(cached)).completedBytes > 0)
            )
              models[index] = { ...cached };
            const resolved = models[index]!;
            if (resolved.browserUnsupportedReason?.startsWith(capabilityErrorPrefix))
              resolved.browserUnsupportedReason = undefined;
            if (inspectEncoding && resolved.encoding) {
              try {
                Object.assign(
                  resolved,
                  restrictCheckpointCapabilities(
                    await inspectEncoding(resolved.encoding),
                    resolved.checkpointCapabilities,
                  ),
                );
                resolved.encodingVerified = true;
              } catch (error) {
                complete = false;
                catalogError = `Could not check chat capabilities: ${errorText(error)}`;
                if (cached?.encodingVerified && JSON.stringify(cached.encoding) === JSON.stringify(resolved.encoding)) {
                  Object.assign(
                    resolved,
                    restrictCheckpointCapabilities(
                      { reasoning: cached.reasoning, supportsTools: cached.supportsTools },
                      resolved.checkpointCapabilities,
                    ),
                  );
                  resolved.encodingVerified = true;
                } else if (!resolved.browserUnsupportedReason) {
                  resolved.browserUnsupportedReason = `${capabilityErrorPrefix} ${errorText(error)}`;
                }
              }
            }
          }
        }),
      );
      for (const cached of previous ?? []) {
        if (
          !models.some((model) => model.identifier === cached.identifier) &&
          (downloads.has(cached.identifier) || (await storedProgress(cached)).completedBytes > 0)
        )
          models.push({
            ...cached,
            browserUnsupportedReason: cached.browserUnsupportedReason?.startsWith(capabilityErrorPrefix)
              ? undefined
              : cached.browserUnsupportedReason,
          });
      }
      catalog = models;
      catalogComplete = complete;
      const cache = await caches.open(MODEL_CACHE);
      await cache.put(
        CATALOG_CACHE_KEY,
        Response.json(
          models.map((model) =>
            model.browserUnsupportedReason?.startsWith(capabilityErrorPrefix)
              ? { ...model, browserUnsupportedReason: undefined }
              : model,
          ),
        ),
      );
      for (const identifier of states.keys()) if (!downloads.has(identifier)) states.delete(identifier);
    } catch (error) {
      catalogComplete = false;
      catalogError = `Could not refresh the model catalog: ${errorText(error)}`;
      if (!catalog) throw error;
    } finally {
      catalogRequest = undefined;
      changed();
    }
  })();
  return catalogRequest;
}

export async function browserModels(): Promise<BrowserModel[]> {
  if (catalog) return catalog;
  initialCatalog ??= (async () => {
    const cache = await caches.open(MODEL_CACHE);
    const stored = await cache.match(CATALOG_CACHE_KEY);
    if (stored) {
      const value: unknown = await stored.json();
      if (
        Array.isArray(value) &&
        value.every((model) => model && typeof model.identifier === "string" && Array.isArray(model.files))
      ) {
        catalog = await Promise.all(
          (value as BrowserModel[]).map(async (model) => {
            const files = browserInferenceFiles(model.files);
            // Keep deletion coverage before dropping old optional inputs from the
            // visible descriptor, including builds that did not record history.
            if (files.length !== model.files.length) await rememberModelFiles(model);
            return {
              ...model,
              files,
              browserUnsupportedReason: browserModelUnsupportedReason({ files }),
              ...restrictCheckpointCapabilities(model, model.checkpointCapabilities),
            };
          }),
        );
        if (typeof document !== "undefined") void refreshCatalog().catch(() => undefined);
        return catalog;
      }
    }
    await refreshCatalog();
    return catalog!;
  })().catch((error: unknown) => {
    initialCatalog = undefined;
    throw error;
  });
  return initialCatalog;
}

export async function findBrowserModel(identifier: string): Promise<BrowserModel> {
  const model = (await browserModels()).find((entry) => entry.identifier === identifier || entry.repoId === identifier);
  if (!model) throw new Error(`Model not found: ${identifier}`);
  return model;
}
export type CachedModelFiles = {
  "config.json": Uint8Array<ArrayBuffer>;
  "tokenizer.json": Uint8Array<ArrayBuffer>;
  "model.safetensors": Blob;
};
export async function cachedModelFiles(model: BrowserModel, checkCancelled?: () => void): Promise<CachedModelFiles> {
  if (model.browserUnsupportedReason) throw new Error(model.browserUnsupportedReason);
  const required = (name: keyof CachedModelFiles) => {
    const file = model.files.find((file) => file.name === name);
    if (!file) throw new Error(`This model is missing ${name}`);
    return file;
  };
  return {
    "config.json": await readModelFile(required("config.json"), checkCancelled),
    "tokenizer.json": await readModelFile(required("tokenizer.json"), checkCancelled),
    "model.safetensors": await openModelFileBlob(required("model.safetensors"), checkCancelled),
  };
}
async function storedProgress(model: BrowserModel) {
  let completedBytes = 0;
  let complete = model.files.length > 0;
  for (const file of model.files) {
    const state = await modelFileStatus(file);
    completedBytes += state.completedBytes;
    complete &&= state.complete;
  }
  return { completedBytes, complete, totalBytes: model.files.reduce((sum, file) => sum + file.sizeBytes, 0) };
}
async function stateFor(model: BrowserModel): Promise<DownloadState> {
  const version = revision(model);
  const existing = states.get(model.identifier);
  if (existing?.revision === version) {
    if (existing.phase !== "Downloading" || downloads.has(model.identifier)) return existing;
    const owner = (await navigator.locks?.query?.())?.held?.some((lock) => lock.name === lockName(model.identifier));
    if (owner) return existing;
    // A crashed owner cannot broadcast its final Paused state.
    states.delete(model.identifier);
  }
  const saved = await browserDb.get<DownloadState>("modelStates", model.identifier);
  const progress = await storedProgress(model);
  const active = (await navigator.locks?.query?.())?.held?.some((lock) => lock.name === lockName(model.identifier));
  // Final state is committed before releasing the writer's lock. An observing
  // tab must not turn a failure, pause, or deletion back into Downloading.
  const terminal =
    saved?.revision === version &&
    (saved.phase === "Error" ||
      saved.phase === "Paused" ||
      (saved.phase === "NotDownloaded" && progress.completedBytes === 0))
      ? saved.phase
      : undefined;
  const state: DownloadState = {
    phase: progress.complete
      ? "Downloaded"
      : terminal
        ? terminal
        : active
          ? "Downloading"
          : progress.completedBytes > 0 || saved?.phase === "Paused" || saved?.phase === "Downloading"
            ? "Paused"
            : "NotDownloaded",
    completedBytes: progress.completedBytes,
    totalBytes: progress.totalBytes,
    revision: version,
    seq: Math.max(saved?.seq ?? 0, nextSequence()),
    ...(saved?.error && saved.revision === version && !progress.complete ? { error: saved.error } : {}),
  };
  const latest = states.get(model.identifier);
  if (latest?.revision === version) return latest;
  states.set(model.identifier, state);
  return state;
}
async function setState(model: BrowserModel, state: Omit<DownloadState, "seq" | "revision">): Promise<void> {
  const next: DownloadState = { ...state, revision: revision(model), seq: nextSequence() };
  await browserDb.put("modelStates", model.identifier, next);
  publishState(model, next);
}
function publishState(model: BrowserModel, next: DownloadState): void {
  states.set(model.identifier, next);
  emit(model, { identifier: model.identifier, kind: "state", ...next, error: next.error ?? null });
  changed();
}

// Chrome reports quota as usage + min(10 GiB, disk size); only below this cap is it a real limit.
const REPORTED_QUOTA_HEADROOM_CAP_BYTES = 10 * 1024 ** 3;

export function fitsReportedQuota(neededBytes: number, estimate: StorageEstimate | undefined): boolean {
  if (estimate?.quota == null || estimate.usage == null) return true;
  return estimate.quota - estimate.usage >= Math.min(neededBytes, REPORTED_QUOTA_HEADROOM_CAP_BYTES);
}

function logDownloadFailure(context: Record<string, unknown>, error: unknown): void {
  const log = (storage?: StorageEstimate) =>
    console.error("[downloads] download failed", { ...context, storage }, error);
  const estimate = navigator.storage?.estimate?.();
  if (!estimate) return log();
  // Not awaited: estimate() can hang on a full disk.
  void estimate.then(log, () => log());
}

async function download(model: BrowserModel, signal: AbortSignal): Promise<void> {
  const totalBytes = model.files.reduce((sum, file) => sum + file.sizeBytes, 0);
  let completedBytes = 0;
  try {
    await rememberModelFiles(model);
    const stored = await storedProgress(model);
    await navigator.storage?.persist?.();
    if (!fitsReportedQuota(totalBytes - stored.completedBytes, await navigator.storage?.estimate?.()))
      throw new Error(
        "Not enough browser storage for this model. Delete a model or free device space, then try again.",
      );
    await setState(model, { phase: "Downloading", completedBytes: stored.completedBytes, totalBytes });
    let lastProgress = 0;
    const files = [...model.files].sort((a, b) => Number(b.name === "config.json") - Number(a.name === "config.json"));
    for (const file of files) {
      signal.throwIfAborted();
      await downloadModelFile(file, signal, (fileBytes) => {
        const now = performance.now();
        if (now - lastProgress < 100 && fileBytes !== file.sizeBytes) return;
        lastProgress = now;
        emit(model, {
          identifier: model.identifier,
          seq: nextSequence(),
          kind: "progress",
          completedBytes: completedBytes + fileBytes,
          totalBytes,
        });
      });
      completedBytes += file.sizeBytes;
      if (file.name === "config.json" && inspectConfig) await inspectConfig(await readModelFile(file));
      await setState(model, { phase: "Downloading", completedBytes, totalBytes });
    }
    signal.throwIfAborted();
    await setState(model, { phase: "Downloaded", completedBytes, totalBytes });
    emit(model, { identifier: model.identifier, seq: nextSequence(), kind: "done" });
  } catch (error) {
    const progress = await storedProgress(model);
    const message = errorText(error);
    if (!signal.aborted)
      logDownloadFailure({ identifier: model.identifier, completedBytes: progress.completedBytes, totalBytes }, error);
    await setState(model, {
      phase: signal.aborted ? "Paused" : "Error",
      completedBytes: progress.completedBytes,
      totalBytes,
      ...(signal.aborted ? {} : { error: message }),
    });
    emit(
      model,
      signal.aborted
        ? { identifier: model.identifier, seq: nextSequence(), kind: "paused" }
        : { identifier: model.identifier, seq: nextSequence(), kind: "error", error: message },
    );
  }
}

async function cancelDownload(model: BrowserModel): Promise<void> {
  modelChannel()?.postMessage({ type: "cancel", identifier: model.identifier });
  const task = downloads.get(model.identifier);
  task?.controller.abort();
  await task?.done;
}

export const browserModelManager = {
  async getModels() {
    modelChannel();
    const models = await browserModels();
    return {
      complete: catalogComplete,
      refreshing: Boolean(catalogRequest),
      error: catalogError,
      models: await Promise.all(
        models.map(async (model): Promise<EngineModel> => {
          const state = await stateFor(model);
          return {
            ...model,
            state: {
              phase: state.phase,
              downloadedKbytes: state.completedBytes / 1024,
              totalKbytes: state.totalBytes / 1024,
              error: state.error,
              seq: state.seq,
            },
          };
        }),
      ),
    };
  },
  async refreshModels() {
    await refreshCatalog();
  },
  async startDownload(identifier: string): Promise<void> {
    modelChannel();
    const model = await findBrowserModel(identifier);
    if (model.browserUnsupportedReason) throw new Error(model.browserUnsupportedReason);
    if (!("gpu" in navigator)) throw new Error("WebGPU is unavailable. Use a browser and device with WebGPU enabled.");
    if (downloads.has(model.identifier)) return;
    const controller = new AbortController();
    const run = async () => {
      if (!navigator.locks) return download(model, controller.signal);
      await navigator.locks.request(lockName(model.identifier), { ifAvailable: true }, async (lock) => {
        if (lock) await download(model, controller.signal);
        else
          emit(model, {
            identifier: model.identifier,
            seq: nextSequence(),
            kind: "locked",
            lockedBy: "another browser tab",
          });
      });
    };
    const done = Promise.resolve()
      .then(run)
      .catch((error: unknown) => {
        // Reporting the error must still work if storage itself rejected the
        // persisted terminal state (for example when quota was exhausted).
        publishState(model, {
          phase: "Error",
          completedBytes: states.get(model.identifier)?.completedBytes ?? 0,
          totalBytes: model.files.reduce((sum, file) => sum + file.sizeBytes, 0),
          revision: revision(model),
          seq: nextSequence(),
          error: errorText(error),
        });
        emit(model, { identifier: model.identifier, seq: nextSequence(), kind: "error", error: errorText(error) });
      })
      .finally(() => downloads.delete(model.identifier));
    downloads.set(model.identifier, { controller, done });
  },
  async pauseDownload(identifier: string): Promise<void> {
    const model = await findBrowserModel(identifier);
    const cancelled = cancelDownload(model);
    if (navigator.locks) await navigator.locks.request(lockName(model.identifier), () => undefined);
    await cancelled;
    states.delete(model.identifier);
    changed();
  },
  async deleteModel(identifier: string): Promise<void> {
    const model = await findBrowserModel(identifier);
    const cancelled = cancelDownload(model);
    const remove = async () => {
      const historyKey = `model-files:${model.identifier}`;
      const previousFiles = (await browserDb.get<BrowserModel["files"]>("modelStates", historyKey)) ?? [];
      for (const file of new Map([...previousFiles, ...model.files].map((file) => [file.url, file])).values())
        await deleteModelFile(file);
      await browserDb.delete("modelStates", historyKey);
      await setState(model, {
        phase: "NotDownloaded",
        completedBytes: 0,
        totalBytes: model.files.reduce((sum, file) => sum + file.sizeBytes, 0),
      });
      emit(model, { identifier: model.identifier, seq: nextSequence(), kind: "deleted" });
    };
    // Queue deletion while the current owner still holds its lock; no restart
    // can enter between cancellation and removal.
    if (navigator.locks) await navigator.locks.request(lockName(model.identifier), remove);
    else {
      await cancelled;
      await remove();
    }
    await cancelled;
  },
  onDownloadEvent(listener: (event: DownloadEvent) => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  onModelsChanged(listener: () => void) {
    changedListeners.add(listener);
    return () => changedListeners.delete(listener);
  },
};
