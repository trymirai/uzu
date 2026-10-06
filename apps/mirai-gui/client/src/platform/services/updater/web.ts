import type { UpdaterService } from ".";
import { noopUnsubscribe } from "../shared/noop";

export const webUpdater: UpdaterService = {
  checkForUpdate: () => Promise.resolve({ currentVersion: "", hasUpdate: false }),
  getUpdateStatus: () => Promise.resolve({ phase: "idle" as const }),
  startUpdateDownload: () => Promise.resolve({ ok: false, error: "unsupported" }),
  applyUpdate: () => Promise.resolve({ ok: false, error: "unsupported" }),
  onUpdateDownloadDone: noopUnsubscribe,
  onUpdateDownloadError: noopUnsubscribe,
};
