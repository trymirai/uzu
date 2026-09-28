import type { ModelCardState, ModelCardStatus } from "./types";

export function deriveModelCardState(state: ModelCardState = { status: "available" }): {
  status: ModelCardStatus;
  isDownloading: boolean;
  isDownloaded: boolean;
  isPaused: boolean;
  isError: boolean;
  progress: number;
} {
  return {
    status: state.status,
    isDownloading: state.status === "downloading" || state.status === "paused",
    isDownloaded: state.status === "downloaded",
    isPaused: state.status === "paused",
    isError: state.status === "error",
    progress: ("progress" in state ? state.progress : undefined) ?? 0,
  };
}
