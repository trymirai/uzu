import { listen } from "@tauri-apps/api/event";
import { invoke } from "../shared/invoke";
import type { ModelCatalog } from "@/types/model-manager";
import type { DownloadEvent, ModelsService } from ".";

// One backend subscription for the app's lifetime, fanned out to listeners.
const listeners = new Set<(event: DownloadEvent) => void>();
const catalogListeners = new Set<() => void>();
let listening: Promise<void> | null = null;

const ensureListening = (): Promise<void> => {
  if (listening) return listening;
  listening = (async () => {
    const stopDownloads = await listen<DownloadEvent>("download-state", ({ payload: event }) => {
      listeners.forEach((listener) => {
        try {
          listener(event);
        } catch (e) {
          // State wiring and toasts are independent; one failing must not starve the other.
          console.error("[downloads] listener failed", { kind: event.kind, identifier: event.identifier }, e);
        }
      });
    });
    try {
      await listen("models-changed", () => catalogListeners.forEach((listener) => listener()));
    } catch (error) {
      stopDownloads();
      throw error;
    }
  })().catch((error) => {
    listening = null;
    throw error;
  });
  return listening;
};

export const tauriModels: ModelsService = {
  getModels: async () => {
    // Subscribe before taking a snapshot so readiness updates cannot fall in between.
    await ensureListening();
    return invoke<ModelCatalog>("chat_models_get");
  },
  refreshModels: () => invoke("chat_models_refresh"),

  startDownload: (repoId) => invoke("download_resume", { repoId }),
  pauseDownload: (repoId) => invoke("download_pause", { repoId }),
  resumeDownload: (repoId) => invoke("download_resume", { repoId }),
  deleteModel: (repoId) => invoke("download_delete", { repoId }),

  onDownloadEvent: (listener) => {
    listeners.add(listener);
    void ensureListening().catch((error) => console.error("[downloads] subscription failed", error));
    return () => listeners.delete(listener);
  },

  onModelsChanged: (listener) => {
    catalogListeners.add(listener);
    void ensureListening().catch((error) => console.error("[models] subscription failed", error));
    return () => catalogListeners.delete(listener);
  },
};
