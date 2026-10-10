import type { ModelCatalog, ModelDownloadPhase } from "@/types/model-manager";

export type DownloadEvent = { identifier: string; seq: number } & (
  | { kind: "state"; phase: ModelDownloadPhase; completedBytes: number; totalBytes: number; error: string | null }
  | { kind: "progress"; completedBytes: number; totalBytes: number | null }
  | { kind: "error"; error: string }
  | { kind: "locked"; lockedBy: string }
  | { kind: "done" | "paused" | "resumed" | "deleted" }
);

export type ModelsService = {
  getModels(): Promise<ModelCatalog>;
  refreshModels(): Promise<void>;
  startDownload(identifier: string): Promise<void>;
  pauseDownload(identifier: string): Promise<void>;
  resumeDownload(identifier: string): Promise<void>;
  deleteModel(identifier: string): Promise<void>;
  onDownloadEvent(listener: (event: DownloadEvent) => void): () => void;
  onModelsChanged(listener: () => void): () => void;
};
