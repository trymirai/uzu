import { useAppStore } from "@/stores/use-app-store";
import { platformInfo } from "@/platform/platform-info";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { attachmentStorage } from "@/features/chat/services/attachment-storage";
import { useEffect } from "react";
import { useSessionWiring } from "./session-wiring";
import { useDownloadWiring } from "./use-download-wiring";
import { useUpdateInitialization } from "./use-update-initialization";
import { useAppTheme } from "./use-app-theme";

export const useAppInitialization = () => {
  useAppTheme();
  useSessionWiring();
  useDownloadWiring();
  useUpdateInitialization(platformInfo.features.autoUpdate);

  useEffect(() => {
    attachmentStorage.cleanup();
    void useModelParamsStore.getState().load();
    void useModelsStore.getState().fetchModels();
    if (platformInfo.isTauri) {
      void useAppStore.getState().fetchVersion();
    }
  }, []);

  // The app is not a browser tab: a reload (F5 or Ctrl+R in WebView2, the page menu everywhere) would drop the reply
  // being streamed, and the page menu offers only browser commands. Fields and selections keep theirs.
  useEffect(() => {
    if (!platformInfo.isTauri) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "F5" || (event.ctrlKey && event.key.toLowerCase() === "r")) event.preventDefault();
    };
    const onContextMenu = (event: MouseEvent) => {
      const field = event.target instanceof Element && event.target.closest("input, textarea, [contenteditable]");
      if (!field && !window.getSelection()?.toString()) event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("contextmenu", onContextMenu);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("contextmenu", onContextMenu);
    };
  }, []);
};
