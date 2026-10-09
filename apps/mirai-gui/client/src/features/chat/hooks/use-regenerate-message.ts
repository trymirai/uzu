import { useCallback } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { resolveModelTools, useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { getPlatform } from "@/platform/platform-singleton";
import type { LlmRunResult } from "@/types/llm-stream";
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
  startStream: (options: StartStreamOptions) => Promise<LlmRunResult | void | null>;
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

      const accepted = await useChatSessionStore.getState().withOperation(
        "running",
        async (signal) => {
          const canceled = () => {
            if (!signal.aborted) return false;
            useChatSessionStore.getState().setLoadingMessage(chatId, null);
            return true;
          };
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
          if (canceled()) return;
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
            if (canceled()) return;
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
            const tools = resolveModelTools(
              useModelParamsStore.getState().getParams(modelId),
              await getPlatform().settings.getModelChatNamingEnabled(),
              useModelsStore.getState().models.find((model) => model.repoId === modelId)?.paramSize,
            );
            if (canceled()) return;
            await startStream({
              signal,
              repoId: modelId,
              messages: messagesForRun,
              messageId,
              chatId,
              ...tools,
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
        },
        chatId,
      );
      if (accepted === null) toast.error("Model is not ready yet, please wait");
    },
    [chatId, globalInstructions, toast, startStream],
  );
};
