import type { UpdateStatus } from "@/types/update";

export type CheckForUpdateResult = {
  currentVersion: string;
  latestVersion?: string;
  hasUpdate: boolean;
  hasDownloadable?: boolean;
  reason?: string;
  source: "gcs" | "error";
};

export type UpdaterService = {
  checkForUpdate(): Promise<CheckForUpdateResult>;
  getUpdateStatus(): Promise<UpdateStatus>;
  startUpdateDownload(version: string): Promise<{ ok: boolean; error?: string }>;
  applyUpdate(): Promise<{ ok: boolean; error?: string }>;
  onUpdateDownloadDone(cb: (payload: { version: string }) => void): () => void;
  onUpdateDownloadError(cb: (payload: { version: string; error: string }) => void): () => void;
};
