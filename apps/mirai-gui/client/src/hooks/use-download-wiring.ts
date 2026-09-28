import { getPlatform } from "@/platform/platform-singleton";
import { useModelsStore } from "@/stores/use-models-store";
import { useEffect } from "react";

export const useDownloadWiring = (enabled: boolean = true) => {
  useEffect(() => {
    if (!enabled) return;
    return getPlatform().models.onDownloadEvent((event) => useModelsStore.getState().applyDownloadEvent(event));
  }, [enabled]);
};
