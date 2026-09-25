import { useAppStore } from "@/stores/useAppStore";
import { getPlatform } from "@/platform/platformSingleton";
import { platformInfo } from "@/platform/platformInfo";
import { useModelParamsStore } from "@/stores/useModelParamsStore";
import { useModelsStore } from "@/stores/useModelsStore";
import { attachmentStorage } from "@/features/chat/services/attachmentStorage";
import { useEffect } from "react";
import { useSessionWiring } from "./sessionWiring";
import { useDownloadWiring } from "./useDownloadWiring";
import { useUpdateInitialization } from "./useUpdateInitialization";

export const useAppInitialization = (enabled: boolean = true) => {
  useSessionWiring(enabled);
  useDownloadWiring(enabled);
  useUpdateInitialization(enabled && platformInfo.features.autoUpdate);

  useEffect(() => {
    if (!enabled) return;
    attachmentStorage.cleanup();
    void useModelParamsStore.getState().load();
    void useModelsStore.getState().fetchModels();
    if (platformInfo.isTauri) {
      void useAppStore.getState().fetchVersion();
      void getPlatform()
        .systemUi.setWindowTheme(useAppStore.getState().isDarkMode)
        .catch(() => {});
    }
  }, [enabled]);
};
