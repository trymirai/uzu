import { useCallback } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { isOtherChatGenerating } from "@/features/runtime/runtime-busy";
import { ejectAndWait } from "@/features/runtime/eject-runtime-session";
import type { ToastApi } from "@/components/ui/toast/use-toast";

type UseModelSwitchParams = {
  chatId: string;
  toast: ToastApi;
  setPendingSwitch: (v: boolean) => void;
  cancelStream: () => Promise<void>;
};

export const useModelSwitch = ({ chatId, toast, setPendingSwitch, cancelStream }: UseModelSwitchParams) => {
  return useCallback(
    async (modelId: string, modelName: string) => {
      const session = useChatSessionStore.getState();
      if (isOtherChatGenerating(session, chatId)) {
        toast.error("A response is already generating in another chat");
        return;
      }

      const resident = useRuntimeSessionStore.getState().residentSession;
      const needsEject = !!resident && resident.repoId !== modelId;

      if (needsEject) {
        try {
          if (session.canStop()) {
            toast.info("Stopping current generation…");
            await cancelStream().catch(() => undefined);
          }

          toast.info("Ejecting previous model…");
          setPendingSwitch(true);
          const closeRes = await ejectAndWait(resident);

          if (!closeRes) {
            setPendingSwitch(false);
            toast.error("Failed to eject previous model");
            return;
          }
        } catch {
          setPendingSwitch(false);
          toast.error("Failed to switch model");
          return;
        }
      }

      const chat = useChatStore.getState();
      chat.suppressAutoSelect(chatId, false);
      chat.setChatModel(chatId, modelId, modelName);
      setPendingSwitch(false);
    },
    [chatId, toast, setPendingSwitch, cancelStream],
  );
};
