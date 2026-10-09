import type { DownloadEvent } from "@/platform/services/models";
import { modelDownloadPhases, type ModelDownloadState } from "@/types/model-manager";
import { toDownloadProgressKbytes } from "@/utils/download-progress";

const progressPatch = (
  current: ModelDownloadState | undefined,
  completedBytes: number,
  totalBytes: number | null,
): Partial<ModelDownloadState> => {
  const { downloadedKbytes, totalKbytes } = toDownloadProgressKbytes(completedBytes, totalBytes);
  const phase = current?.phase;
  // Progress ticks keep arriving briefly after pause, completion and a failure.
  // Only bytes that actually grew may pull an errored download back to active.
  const grew = downloadedKbytes > (current?.downloadedKbytes ?? 0);
  const becomesActive =
    phase !== modelDownloadPhases.downloaded &&
    phase !== modelDownloadPhases.paused &&
    (phase !== modelDownloadPhases.error || grew);
  const clearsError = phase === modelDownloadPhases.locked || phase === modelDownloadPhases.error;

  return {
    downloadedKbytes,
    ...(totalKbytes ? { totalKbytes } : {}),
    ...(becomesActive ? { phase: modelDownloadPhases.downloading } : {}),
    ...(becomesActive && clearsError ? { error: "" } : {}),
  };
};

export const downloadStatePatch = (
  current: ModelDownloadState | undefined,
  event: DownloadEvent,
): Partial<ModelDownloadState> | null => {
  switch (event.kind) {
    case "state":
      return {
        ...toDownloadProgressKbytes(event.completedBytes, event.totalBytes),
        phase: event.phase,
        error: event.error ?? "",
      };
    case "progress":
      return progressPatch(current, event.completedBytes, event.totalBytes);
    case "done":
      return { phase: modelDownloadPhases.downloaded, error: "" };
    case "error":
      return { phase: modelDownloadPhases.error, error: event.error };
    case "locked":
      return { phase: modelDownloadPhases.locked, error: "" };
    case "paused":
      return { phase: modelDownloadPhases.paused };
    case "resumed":
      // A terminal error stays visible until progress proves the download recovered.
      return current?.phase === modelDownloadPhases.error
        ? null
        : { phase: modelDownloadPhases.downloading, error: "" };
    case "deleted":
      return { phase: modelDownloadPhases.notDownloaded, downloadedKbytes: 0, totalKbytes: 0, error: "" };
  }
};

type WithTotalKbytes = {
  totalKbytes: number;
};

// A catalog snapshot may not know the size yet; keep the one already seen.
export const preserveTotalKbytes = <TState extends WithTotalKbytes>(
  incomingState: TState,
  previousState?: TState,
): TState => {
  const fallbackTotalKbytes =
    previousState?.totalKbytes && previousState.totalKbytes > 0 ? previousState.totalKbytes : undefined;
  const effectiveTotalKbytes =
    incomingState.totalKbytes && incomingState.totalKbytes > 0 ? incomingState.totalKbytes : fallbackTotalKbytes;

  return typeof effectiveTotalKbytes === "number"
    ? { ...incomingState, totalKbytes: effectiveTotalKbytes }
    : incomingState;
};
