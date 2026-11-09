import type { Model, EngineCallback } from "@uzu/wasm";
import type { DownloadEvent, ModelsService } from "@/platform/services/models";
import type { ModelDownloadState } from "@/types/model-manager";
import { toDownloadProgressKbytes } from "@/utils/download-progress";
import { errorText } from "@/utils/error-text";
import { getUzuRuntime, type UzuRuntime } from "../uzu-runtime";
import { describeModel, describeState, type CatalogModel } from "./model-adapter";

type Snapshot = { models: CatalogModel[]; complete: boolean; refreshing: boolean };

// This adapter only translates the existing Rust API into the catalog UI contract.
// Uzu owns registry requests, downloads, locks, persistence and OPFS paths.
export function createModelManager(loadRuntime: () => Promise<UzuRuntime> = getUzuRuntime): ModelsService & {
  modelPath(identifier: string): Promise<string | null>;
} {
  const listeners = new Set<() => void>();
  const downloadListeners = new Set<(event: DownloadEvent) => void>();
  const lastStates = new Map<string, ModelDownloadState>();
  const models = new Map<string, Model>();
  const active = new Map<string, Promise<void>>();
  const errors = new Map<string, string>();
  let runtime: Promise<UzuRuntime> | undefined;
  let callback: EngineCallback | undefined;
  let refreshing = false;
  let sequence = 0;
  let snapshotVersion = 0;

  const notify = () => {
    listeners.forEach((listener) => listener());
    // Event-only consumers do not need to poll the catalog to receive native updates.
    if (listeners.size === 0 && downloadListeners.size > 0) void readSnapshot().catch(() => {});
  };
  const emit = (event: DownloadEvent) => downloadListeners.forEach((listener) => listener(event));

  function initialize() {
    runtime ??= loadRuntime()
      .then(async (uzu) => {
        callback = uzu.api.EngineCallback.create(notify);
        try {
          await uzu.engine.registerCallback(callback);
        } catch (cause) {
          callback.free();
          callback = undefined;
          throw cause;
        }
        return uzu;
      })
      .catch((cause: unknown) => {
        runtime = undefined;
        throw cause;
      });
    return runtime;
  }

  async function readSnapshot(): Promise<Snapshot> {
    const version = ++snapshotVersion;
    const initialSequence = sequence;
    const { api, engine } = await initialize();
    const nativeModels = await engine.modelsForChat;
    const descriptors = nativeModels.flatMap((model) => {
      try {
        return model.isDownloadable ? [describeModel(model, api)] : [];
      } finally {
        if (model.isDownloadable && !models.has(model.identifier)) models.set(model.identifier, model);
        else model.free();
      }
    });
    const states = await engine.downloadStates;
    const normalized = new Map(Object.entries(states).map(([id, state]) => [id, describeState(state, api)]));
    return {
      complete: true,
      refreshing,
      models: descriptors.map((model) => {
        const state: ModelDownloadState = {
          phase: "Initializing",
          ...toDownloadProgressKbytes(
            0,
            model.files.reduce((total, file) => total + file.sizeBytes, 0),
          ),
          ...normalized.get(model.identifier),
          ...(errors.has(model.identifier) ? ({ phase: "Error", error: errors.get(model.identifier) } as const) : {}),
          seq: initialSequence,
        };
        // An older asynchronous read must never publish over a newer native snapshot.
        if (version !== snapshotVersion) return { ...model, state };
        const previous = lastStates.get(model.identifier);
        if (
          previous &&
          previous.phase === state.phase &&
          previous.downloadedKbytes === state.downloadedKbytes &&
          previous.totalKbytes === state.totalKbytes &&
          previous.error === state.error
        ) {
          return { ...model, state: previous };
        }
        const identifier = model.repoId ?? model.identifier;
        state.seq = ++sequence;
        lastStates.set(model.identifier, state);
        emit({
          kind: "state",
          identifier,
          seq: state.seq,
          phase: state.phase,
          completedBytes: state.downloadedKbytes * 1024,
          totalBytes: state.totalKbytes * 1024,
          error: state.error ?? null,
        });
        if (previous && previous.phase !== state.phase) {
          const terminal =
            state.phase === "Downloaded"
              ? "done"
              : state.phase === "Paused"
                ? "paused"
                : state.phase === "NotDownloaded"
                  ? "deleted"
                  : null;
          const finishedDownload = previous.phase === "Downloading" || active.has(model.identifier);
          if (terminal && (terminal !== "done" || finishedDownload))
            emit({ kind: terminal, identifier, seq: (state.seq = ++sequence) });
          else if (state.phase === "Error")
            emit({ kind: "error", identifier, seq: (state.seq = ++sequence), error: state.error ?? "Download failed" });
          else if (state.phase === "Locked")
            emit({ kind: "locked", identifier, seq: (state.seq = ++sequence), lockedBy: "another browser tab" });
        }
        return { ...model, state };
      }),
    };
  }

  // Each notification must read fresh native state, even if an older read is pending.
  const getModels = readSnapshot;

  async function findModel(identifier: string) {
    await initialize();
    let model = models.get(identifier) ?? [...models.values()].find((entry) => entry.repoIds.includes(identifier));
    if (!model) {
      await getModels();
      model = models.get(identifier) ?? [...models.values()].find((entry) => entry.repoIds.includes(identifier));
    }
    if (!model) throw new Error(`Model not found: ${identifier}`);
    return model;
  }

  return {
    getModels,
    async refreshModels() {
      const { engine } = await initialize();
      refreshing = true;
      notify();
      try {
        const refreshed = await engine.refreshModels();
        for (const model of refreshed) model.free();
      } finally {
        refreshing = false;
        notify();
      }
    },
    async startDownload(identifier: string) {
      const { engine } = await initialize();
      const model = await findModel(identifier);
      const id = model.identifier;
      if (active.has(id)) return;
      errors.delete(id);
      // Register the operation before awaiting: double clicks cannot start two streams.
      const operation = (async () => {
        const stream = await engine.download(model);
        try {
          notify();
          while (true) {
            const state = await stream.next();
            if (!state) break;
            state.free();
            notify();
          }
        } finally {
          stream.free();
        }
      })()
        .catch((cause: unknown) => {
          errors.set(id, errorText(cause));
        })
        .finally(() => {
          active.delete(id);
          notify();
        });
      active.set(id, operation);
    },
    async resumeDownload(identifier: string) {
      return this.startDownload(identifier);
    },
    async pauseDownload(identifier: string) {
      const { engine } = await initialize();
      const model = await findModel(identifier);
      const downloader = engine.downloader(model);
      try {
        await downloader.pause();
        await active.get(model.identifier);
      } finally {
        downloader.free();
        notify();
      }
    },
    async deleteModel(identifier: string) {
      const { engine } = await initialize();
      const model = await findModel(identifier);
      const downloader = engine.downloader(model);
      try {
        await downloader.delete();
        await active.get(model.identifier);
        errors.delete(model.identifier);
      } finally {
        downloader.free();
        notify();
      }
    },
    async modelPath(identifier: string) {
      const { engine } = await initialize();
      return engine.modelPath(await findModel(identifier));
    },
    onModelsChanged(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    onDownloadEvent(listener: (event: DownloadEvent) => void) {
      downloadListeners.add(listener);
      return () => {
        downloadListeners.delete(listener);
      };
    },
  };
}

export const browserModelManager = createModelManager();
