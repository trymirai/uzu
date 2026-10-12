import { getPlatform } from "@/platform/platform-singleton";
import { useModelsStore } from "@/stores/use-models-store";
import { useEffect } from "react";

export const useDownloadWiring = (enabled: boolean = true) => {
  useEffect(() => {
    if (!enabled) return;
    const models = getPlatform().models;
    const stopDownloads = models.onDownloadEvent((event) => useModelsStore.getState().applyDownloadEvent(event));
    const stopCatalog = models.onModelsChanged(() => void useModelsStore.getState().fetchModels());
    return () => {
      stopDownloads();
      stopCatalog();
    };
  }, [enabled]);
};
