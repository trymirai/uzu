import { getPlatform } from "@/platform/platform-singleton";
import { UpdateStatusPhase } from "@/types/update";
import { create } from "zustand";

export const UpdatePhase = {
  ...UpdateStatusPhase,
  Available: "available",
} as const;
export type UpdatePhase = (typeof UpdatePhase)[keyof typeof UpdatePhase];

export type UpdateDownloadStatus =
  | { phase: typeof UpdatePhase.Idle; downloadError?: string }
  | { phase: typeof UpdatePhase.Available; version: string }
  | { phase: typeof UpdatePhase.Downloading; version: string }
  | { phase: typeof UpdatePhase.Downloaded; version: string; applyError?: string };

type UpdateState = {
  status: UpdateDownloadStatus;
  _downloadListenersCleanup: (() => void) | null;

  initUpdateCheck(): Promise<void>;
  checkForUpdate(): Promise<void>;
  startDownload(version: string): Promise<void>;
  watchDownload(version: string): void;
  _reconcileDownload(version: string): Promise<void>;
  applyUpdate(): Promise<void>;
  dismiss(): void;
};

export const useUpdateStore = create<UpdateState>()((set, get) => ({
  status: { phase: UpdatePhase.Idle },
  _downloadListenersCleanup: null,

  async initUpdateCheck() {
    await get().checkForUpdate();
    try {
      const status = await getPlatform().updater.getUpdateStatus();
      // Hydrate every backend phase, not only Downloaded: a webview reload while
      // a download is in flight must resume watching, or the store stays stuck
      // showing "downloading" forever with no listeners.
      if (status.phase === UpdateStatusPhase.Downloaded) {
        set({ status: { phase: UpdatePhase.Downloaded, version: status.version } });
      } else if (status.phase === UpdateStatusPhase.Downloading) {
        get().watchDownload(status.version);
        await get()._reconcileDownload(status.version);
      }
    } catch (e) {
      console.warn("[update] status hydration failed", e);
    }
  },

  async checkForUpdate() {
    try {
      const updater = getPlatform().updater;
      const res = await updater.checkForUpdate();

      if (!res.hasUpdate || !res.latestVersion || !res.hasDownloadable) return;

      const controllerStatus = await updater.getUpdateStatus();
      const updateAlreadyInFlight = controllerStatus.phase !== UpdateStatusPhase.Idle;
      if (updateAlreadyInFlight) return;

      set({ status: { phase: UpdatePhase.Available, version: res.latestVersion } });
    } catch {
      // Offline or a bad manifest simply means no update to offer.
    }
  },

  watchDownload(version) {
    const updater = getPlatform().updater;

    const prevCleanup = get()._downloadListenersCleanup;
    if (prevCleanup) prevCleanup();

    set({ status: { phase: UpdatePhase.Downloading, version } });

    const cleanup = () => {
      offDone();
      offError();
      set({ _downloadListenersCleanup: null });
    };

    const offDone = updater.onUpdateDownloadDone(({ version: v }) => {
      if (v !== version) return;
      cleanup();
      set({ status: { phase: UpdatePhase.Downloaded, version } });
    });

    const offError = updater.onUpdateDownloadError(({ version: v, error }) => {
      if (v !== version) return;
      cleanup();
      set({ status: { phase: UpdatePhase.Idle, downloadError: error } });
    });

    set({ _downloadListenersCleanup: cleanup });
  },

  // listen() is async, so an event can fire before the handler is live and be
  // lost; every watcher rechecks the backend snapshot to catch that.
  async _reconcileDownload(version) {
    if (get().status.phase !== UpdatePhase.Downloading) return;
    try {
      const status = await getPlatform().updater.getUpdateStatus();
      if (status.phase === UpdateStatusPhase.Downloaded && get().status.phase === UpdatePhase.Downloading) {
        get()._downloadListenersCleanup?.();
        set({ status: { phase: UpdatePhase.Downloaded, version } });
      }
    } catch {
      // Reconciliation is a safety net for a missed event; the listeners still stand.
    }
  },

  async startDownload(version) {
    const updater = getPlatform().updater;

    get().watchDownload(version);

    const result = await updater.startUpdateDownload(version);
    if (!result.ok) {
      get()._downloadListenersCleanup?.();
      set({ status: { phase: UpdatePhase.Idle, downloadError: result.error } });
      return;
    }

    await get()._reconcileDownload(version);
  },

  async applyUpdate() {
    const current = get().status;
    if (current.phase !== UpdatePhase.Downloaded) return;
    const version = current.version;
    try {
      const result = await getPlatform().updater.applyUpdate();
      if (!result.ok) {
        set({
          status: { phase: UpdatePhase.Downloaded, version, applyError: result.error || "Failed to apply update" },
        });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to apply update";
      set({ status: { phase: UpdatePhase.Downloaded, version, applyError: msg } });
    }
  },

  dismiss() {
    set({ status: { phase: UpdatePhase.Idle } });
  },
}));
