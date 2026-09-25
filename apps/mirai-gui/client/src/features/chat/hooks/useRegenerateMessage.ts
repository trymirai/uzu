import { useCallback } from "react";
import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { useChatStore } from "@/stores/useChatStore";
import type { RuntimeSessionRef } from "@/types/session";
import { Roles } from "../types";
import { attachmentStorage } from "../services/attachmentStorage";
import { getChatRunBlockMessage, prepareChatModelForRun } from "../services/chatRunPreflight";
import { buildChatRunInput } from "../services/buildChatRunInput";
import { patchActiveVersion, projectAssistantVersion } from "../services/regenerateVersions";
import type { StartStreamOptions } from "./useLlmStream";
import type { ToastApi } from "@/ui-kit";

type UseRegenerateMessageParams = {
  chatId: string;
  globalInstructions: string | null | undefined;
  toast: ToastApi;
  startStream: (options: StartStreamOptions) => Promise<void | null>;
};

const waitUntilCanRun = (target: RuntimeSessionRef, attempts: number): Promise<boolean> =>
  new Promise((resolve) => {
    const attempt = (left: number) => {
      const ok = useChatSessionStore.getState().canRun(target);
      if (ok) resolve(true);
      else if (left <= 0) resolve(false);
      else window.setTimeout(() => attempt(left - 1), 120);
    };
    attempt(attempts);
  });

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

      const canRunNow = await waitUntilCanRun(target, 30);
      if (!canRunNow) {
        toast.error("Model is not ready yet, please wait");
        return;
      }

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

      void startStream({
        repoId: modelId,
        messages: messagesForRun,
        messageId,
        chatId,
        onDone: () => setLoadingMessage(chatId, null),
        updateText,
        onError: (id, errorText) => {
          const current = useChatStore.getState().messages.find((m) => m.id === id);
          const nextVersions = patchActiveVersion(current?.versions, { text: "", error: errorText });
          useChatStore
            .getState()
            .updateMessage(
              id,
              nextVersions.length > 0 ? { text: "", versions: nextVersions } : { text: "", error: errorText },
            );
          setLoadingMessage(chatId, null);
        },
        onFinishReason: (reason) => {
          if (reason === "ContextLimitReached") {
            toast.warning("Conversation reached the model's context limit; reply may be truncated");
          } else if (reason === "Length") {
            toast.warning("Reply hit the output length limit and may be truncated");
          }
        },
      });
    },
    [chatId, globalInstructions, toast, startStream],
  );
};
