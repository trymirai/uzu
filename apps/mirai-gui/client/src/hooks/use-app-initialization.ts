import { useAppStore } from "@/stores/use-app-store";
import { getPlatform } from "@/platform/platform-singleton";
import { platformInfo } from "@/platform/platform-info";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { attachmentStorage } from "@/features/chat/services/attachment-storage";
import { useEffect } from "react";
import { useSessionWiring } from "./session-wiring";
import { useDownloadWiring } from "./use-download-wiring";
import { useUpdateInitialization } from "./use-update-initialization";

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
