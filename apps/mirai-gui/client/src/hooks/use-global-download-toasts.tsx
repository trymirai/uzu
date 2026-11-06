import { InlineModelName } from "@/components/model-name";
import { useToast } from "@/components/ui/toast/use-toast";
import { getPlatform } from "@/platform/platform-singleton";
import { navigationRequestTypes } from "@/platform/platform-client";
import { useModelsStore } from "@/stores/use-models-store";
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
            toast.success(
              <>
                <InlineModelName name={name} /> is ready
              </>,
              { description: "Click to start a chat", onClick: () => openChatForModel(event.identifier) },
            );
            return;
          case "error":
            toast.error(
              <>
                Couldn't download <InlineModelName name={name} />
              </>,
              { description: event.error },
            );
            return;
          case "locked":
            toast.info(
              <>
                <InlineModelName name={name} /> is downloading in another app
              </>,
              { description: `Held by ${event.lockedBy}` },
            );
            return;
        }
      }),
    [toast],
  );
};
