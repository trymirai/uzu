import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { UpdateStatus } from "@/types/update";
import type { CheckForUpdateResult, UpdaterService } from ".";
import { createKeyedListeners } from "../shared/keyedListeners";

type RustCheck = {
  currentVersion: string;
  latestVersion?: string;
  hasUpdate: boolean;
  hasDownloadable: boolean;
  reason?: string;
  source: "gcs" | "error";
};

const state = {
  listeners: createKeyedListeners(),
  listening: false,
};

const ensureListening = (): void => {
  if (state.listening) return;
  state.listening = true;
  void listen<{ version: string }>("update-download-done", (e) => state.listeners.emit("done", e.payload));
  void listen<{ version: string; error: string }>("update-download-error", (e) =>
    state.listeners.emit("error", e.payload),
  );
};

const on = (kind: "done" | "error", cb: (payload: never) => void): (() => void) => {
  ensureListening();
  return state.listeners.on(kind, cb);
};

export const tauriUpdater: UpdaterService = {
  checkForUpdate: async (): Promise<CheckForUpdateResult> => {
    const res = await invoke<RustCheck>("update_check");
    return { ...res };
  },
  getUpdateStatus: () => invoke<UpdateStatus>("update_status"),
  startUpdateDownload: (version) => invoke<{ ok: boolean; error?: string }>("update_download", { version }),
  applyUpdate: () => invoke<{ ok: boolean; error?: string }>("update_apply"),
  onUpdateDownloadDone: (cb) => on("done", cb),
  onUpdateDownloadError: (cb) => on("error", cb),
};
