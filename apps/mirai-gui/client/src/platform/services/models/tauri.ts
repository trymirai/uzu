import { listen } from "@tauri-apps/api/event";
import { invoke } from "../shared/invoke";
import type { EngineModel } from "@/types/modelManager";
import type { DownloadEvent, ModelsService } from ".";

// One backend subscription for the app's lifetime, fanned out to listeners.
const listeners = new Set<(event: DownloadEvent) => void>();
let listening = false;

const ensureListening = (): void => {
  if (listening) return;
  listening = true;
  void listen<DownloadEvent>("download-state", ({ payload: event }) => {
    listeners.forEach((listener) => {
      try {
        listener(event);
      } catch (e) {
        // State wiring and toasts are independent; one failing must not starve the other.
        console.error("[downloads] listener failed", { kind: event.kind, identifier: event.identifier }, e);
      }
    });
  });
};

export const tauriModels: ModelsService = {
  getModels: () => invoke<EngineModel[]>("chat_models_get"),

  startDownload: (repoId) => invoke("download_resume", { repoId }),
  pauseDownload: (repoId) => invoke("download_pause", { repoId }),
  resumeDownload: (repoId) => invoke("download_resume", { repoId }),
  deleteModel: (repoId) => invoke("download_delete", { repoId }),

  onDownloadEvent: (listener) => {
    ensureListening();
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};
