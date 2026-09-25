import { useToast } from "@/ui-kit";
import { getPlatform } from "@/platform/platformSingleton";
import { navigationRequestTypes } from "@/platform/PlatformClient";
import { useModelsStore } from "@/stores/useModelsStore";
import { useEffect } from "react";

const isAppFocused = (): boolean => document.visibilityState === "visible" && document.hasFocus();

const modelName = (identifier: string): string =>
  useModelsStore.getState().models.find((model) => model.repoId === identifier)?.name ?? identifier;

const openChatForModel = (identifier: string): void => {
  window.dispatchEvent(new CustomEvent(navigationRequestTypes.openChatForModel, { detail: { identifier } }));
};

export const useGlobalDownloadToasts = (): void => {
  const toast = useToast();

  useEffect(
    () =>
      getPlatform().models.onDownloadEvent((event) => {
        // The backend raises a system notification when the window is out of sight.
        if (!isAppFocused()) return;
        const name = modelName(event.identifier);
        switch (event.kind) {
          case "done":
            toast.success(`Downloaded ${name}`, { onClick: () => openChatForModel(event.identifier) });
            return;
          case "error":
            toast.error(`Failed to download ${name}: ${event.error}`);
            return;
          case "locked":
            toast.info(`${name} download is locked by another process (${event.lockedBy}).`);
            return;
        }
      }),
    [toast],
  );
};
