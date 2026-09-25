import { getPlatform } from "@/platform/platformSingleton";
import { useModelsStore } from "@/stores/useModelsStore";
import { useEffect } from "react";

export const useDownloadWiring = (enabled: boolean = true) => {
  useEffect(() => {
    if (!enabled) return;
    return getPlatform().models.onDownloadEvent((event) => useModelsStore.getState().applyDownloadEvent(event));
  }, [enabled]);
};
