import { useCallback } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { Roles } from "../types";
import { attachmentStorage } from "../services/attachment-storage";
import { getChatRunBlockMessage, prepareChatModelForRun } from "../services/chat-run-preflight";
import { buildChatRunInput } from "../services/build-chat-run-input";
import { patchActiveVersion, projectAssistantVersion } from "../services/regenerate-versions";
import type { StartStreamOptions } from "./use-llm-stream";
import type { ToastApi } from "@/components/ui/toast/use-toast";

type UseRegenerateMessageParams = {
  chatId: string;
  globalInstructions: string | null | undefined;
  toast: ToastApi;
  startStream: (options: StartStreamOptions) => Promise<void | null>;
};

export const useRegenerateMessage = ({
  chatId,
  globalInstructions,
  toast,
  startStream,
}: UseRegenerateMessageParams) => {
  return useCallback(
    async (messageId: string, modelId: string, modelName: string) => {
      const target = { repoId: modelId } as const;
      const ready = await prepareChatModelForRun({ chatId, modelId }, target, {
        onEjectStart: () => toast.info("Ejecting previous model…"),
      });
      if (!ready.ok) {
        toast.error(getChatRunBlockMessage(ready.reason));
        return;
      }

      const state = useChatStore.getState();
      const messages = state.messages;
      const targetIndex = messages.findIndex((m) => m.id === messageId);
      if (targetIndex < 0) return;
      const targetMessage = messages[targetIndex];
      if (!targetMessage || targetMessage.sender !== Roles.Assistant) return;

      const userIndex = messages
        .slice(0, targetIndex)
        .reduceRight((found, m, i) => (found !== -1 ? found : m.sender === Roles.User ? i : -1), -1);
      if (userIndex < 0) return;

      const userMessage = messages[userIndex];
      if (!userMessage) return;
      const userAttachments = userMessage.attachmentIds ? attachmentStorage.getFiles(userMessage.attachmentIds) : [];
      const { messages: messagesForRun } = buildChatRunInput({
        history: messages.slice(0, userIndex),
        prompt: userMessage.text,
        attachments: userAttachments,
        globalInstructions,
      });

      const accepted = await useChatSessionStore.getState().withOperation("running", async () => {
        const newVersions = projectAssistantVersion(targetMessage, modelId, modelName, userMessage.attachmentIds);
        const regeneratePatch = {
          text: "",
          modelId,
          modelName,
          versions: newVersions,
          currentVersionIndex: newVersions.length - 1,
        };
        state.updateMessage(messageId, regeneratePatch);
        await state.persistMessagePatch(chatId, messageId, regeneratePatch);
        const { setLoadingMessage } = useChatSessionStore.getState();
        setLoadingMessage(chatId, messageId);

        const updateText = (id: string, updatedText: string) => {
          const current = useChatStore.getState().messages.find((m) => m.id === id);
          useChatStore.getState().updateMessage(id, {
            text: updatedText,
            versions: patchActiveVersion(current?.versions, {
              text: updatedText,
              attachmentIds: userMessage.attachmentIds,
            }),
          });
        };

        const failVersion = (errorText: string) => {
          const current = useChatStore.getState().messages.find((m) => m.id === messageId);
          const nextVersions = patchActiveVersion(current?.versions, { text: "", error: errorText });
          useChatStore
            .getState()
            .updateMessage(
              messageId,
              nextVersions.length > 0 ? { text: "", versions: nextVersions } : { text: "", error: errorText },
            );
          setLoadingMessage(chatId, null);
        };

        try {
          await startStream({
            repoId: modelId,
            messages: messagesForRun,
            messageId,
            chatId,
            onDone: () => setLoadingMessage(chatId, null),
            updateText,
            onError: (_id, errorText) => failVersion(errorText),
            onFinishReason: (reason) => {
              if (reason === "ContextLimitReached") {
                toast.warning("Conversation reached the model's context limit; reply may be truncated");
              } else if (reason === "Length") {
                toast.warning("Reply hit the output length limit and may be truncated");
              }
            },
          });
        } catch (e) {
          failVersion(`Error: ${e instanceof Error ? e.message : String(e)}`);
        }
      });
      if (accepted === null) toast.error("Model is not ready yet, please wait");
    },
    [chatId, globalInstructions, toast, startStream],
  );
};
