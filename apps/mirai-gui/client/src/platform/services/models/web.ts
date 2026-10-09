import type { ModelsService } from ".";
import { noopUnsubscribe } from "../shared/noop";

const unavailable = () => Promise.reject(new Error("Not available on web"));

export const webModels: ModelsService = {
  getModels: () => Promise.resolve({ models: [], complete: true, refreshing: false }),
  refreshModels: () => Promise.resolve(),
  startDownload: unavailable,
  pauseDownload: unavailable,
  resumeDownload: unavailable,
  deleteModel: unavailable,
  onDownloadEvent: noopUnsubscribe,
  onModelsChanged: noopUnsubscribe,
};
