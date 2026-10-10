import { useCallback } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { resolveModelTools, useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { getPlatform } from "@/platform/platform-singleton";
import type { LlmRunResult } from "@/types/llm-stream";
import type { AttachedFile } from "@/types/files";
import type { Message } from "@/types/message";
import { Roles } from "../types";
import { attachmentStorage } from "../services/attachment-storage";
import { ChatRunBlockReason } from "../types";
import { getChatRunBlockMessage, prepareChatModelForRun } from "../services/chat-run-preflight";
import { buildChatRunInput } from "../services/build-chat-run-input";
import type { ChatComposerState } from "./use-chat-composer-state";
import type { StartStreamOptions } from "./use-llm-stream";
import type { ToastApi } from "@/components/ui/toast/use-toast";

type UseSendChatMessageParams = {
  chatId: string;
  composer: ChatComposerState;
  globalInstructions: string | null | undefined;
  runStatusBlockReason: ChatRunBlockReason | null;
  toast: ToastApi;
  setScrollTargetId: (id: string | null) => void;
  startStream: (options: StartStreamOptions) => Promise<LlmRunResult | void | null>;
};

type MessageEdit = {
  messageId: string;
  onSaved: () => void;
};

export const useSendChatMessage = (params: UseSendChatMessageParams) => {
  const { chatId, composer, globalInstructions, runStatusBlockReason, toast, setScrollTargetId, startStream } = params;

  return useCallback(
    async (message: string, edit?: MessageEdit): Promise<void> => {
      const session = useChatSessionStore.getState();
      if (edit && useChatStore.getState().currentChatId !== chatId) return;
      useChatStore.getState().setCurrentChatId(chatId);

      const storeState = useChatStore.getState();
      const modelId = storeState.chatModels[chatId]?.modelId;
      if (!modelId) throw new Error("Model is not selected");

      // Captured before any await: the user may open another chat while this
      // one waits for an eject or a write, and the store then holds that chat.
      let history = storeState.messages;
      if (edit) {
        const original = history.find((m) => m.id === edit.messageId && m.sender === Roles.User);
        if (!original) throw new Error("The message is no longer in this chat");
        if (!message.trim() && !original.attachmentIds?.length) return;
      }

      if (runStatusBlockReason !== null) {
        if (runStatusBlockReason === ChatRunBlockReason.OtherChatGenerating) {
          toast.error("A response is already generating in another chat");
        }
        return;
      }

      const target = { repoId: modelId } as const;
      const ready = await prepareChatModelForRun({ chatId, modelId }, target, {
        onEjectStart: () => toast.info("Ejecting previous model…"),
      });
      if (!ready.ok) {
        toast.error(getChatRunBlockMessage(ready.reason));
        return;
      }

      const canRunNow = useChatSessionStore.getState().canRun(target);
      if (!canRunNow) {
        toast.error("Model is not ready yet, please wait");
        return;
      }

      const accepted = await session.withOperation(
        "running",
        async (signal) => {
          const canceled = () => {
            if (!signal.aborted) return false;
            session.setLoadingMessage(chatId, null);
            return true;
          };
          let userMessage: Message;
          let filesToSend: AttachedFile[];
          if (edit) {
            // Commit the replacement and truncation together before changing the
            // view. Use the saved history even if the user navigates while saving.
            const messages = await useChatStore.getState().editUserMessage(chatId, edit.messageId, message, signal);
            if (canceled()) return;
            userMessage = messages[messages.length - 1]!;
            history = messages.slice(0, -1);
            filesToSend = attachmentStorage.getFiles(userMessage.attachmentIds ?? []);
            edit.onSaved();
          } else {
            filesToSend = [...composer.attachedFiles];
            const fileIds = filesToSend.map((file) => {
              attachmentStorage.saveFile(file);
              return file.id;
            });
            userMessage = useChatStore.getState().addMessageTo(chatId, {
              text: message,
              sender: Roles.User,
              attachmentIds: fileIds.length > 0 ? fileIds : undefined,
            });
            composer.clear(message);
          }
          const attachmentIds = userMessage.attachmentIds;
          if (useChatStore.getState().currentChatId === chatId) setScrollTargetId(userMessage.id);

          // Created before the first await so Stop during title generation has a reply to remove.
          const placeholder = useChatStore.getState().addMessageTo(chatId, {
            text: "",
            sender: Roles.Assistant,
            modelId: modelId || undefined,
            modelName: storeState.chatModels[chatId]?.modelName || undefined,
            perf: {},
            attachmentIds,
          });
          session.setLoadingMessage(chatId, placeholder.id);

          const failReply = async (errText: string) => {
            if (canceled()) return;
            session.setLoadingMessage(chatId, null);
            useChatStore.getState().updateMessage(placeholder.id, { text: "", error: errText, attachmentIds });
            await useChatStore
              .getState()
              .persistMessage(chatId, placeholder.id, { ...placeholder, text: "", error: errText })
              .catch((err) => console.error("[storage] failed to persist error message", err));
          };

          try {
            if (!edit) await useChatStore.getState().persistMessage(chatId, userMessage.id, userMessage);
            if (canceled()) return;
            await useChatStore.getState().persistMessage(chatId, placeholder.id, placeholder);
            if (canceled()) return;

            const { messages: messagesForRun } = buildChatRunInput({
              history,
              prompt: message,
              attachments: filesToSend,
              globalInstructions,
            });

            const { modelChatNamingEnabled, dateTimeToolEnabled, chartToolEnabled } = resolveModelTools(
              useModelParamsStore.getState().getParams(modelId),
              await getPlatform().settings.getModelChatNamingEnabled(),
              useModelsStore.getState().models.find((model) => model.repoId === modelId)?.paramSize,
            );
            if (canceled()) return;

            if (!modelChatNamingEnabled) {
              const titleGenResult = await useChatStore.getState().generateChatTitle(chatId, message);

              if (useChatSessionStore.getState().consumeTitleGenAbort(chatId)) {
                // The placeholder is already on disk; remove it after Stop so it
                // cannot return as a blank bubble after a reload.
                session.setCanceledMessage(chatId, null);
                session.setLoadingMessage(chatId, null);
                await useChatStore.getState().discardMessage(chatId, placeholder.id);
                return;
              }

              if (!titleGenResult.ok && titleGenResult.error) {
                await failReply(`Error: ${titleGenResult.error}`);
                return;
              }
            }
            if (canceled()) return;

            const result = await startStream({
              signal,
              repoId: modelId,
              messages: messagesForRun,
              modelChatNamingEnabled,
              dateTimeToolEnabled,
              chartToolEnabled,
              messageId: placeholder.id,
              chatId,
              updateText: (id: string, updatedText: string) => {
                useChatStore.getState().updateMessage(id, { text: updatedText, attachmentIds });
              },
              onDone: () => session.setLoadingMessage(chatId, null),
              onError: (id, errorText) => {
                const current = useChatStore.getState().messages.find((m) => m.id === id);
                useChatStore.getState().updateMessage(id, {
                  text: "",
                  error: errorText,
                  attachmentIds: current?.attachmentIds,
                });
                void useChatStore.getState().persistMessageError(chatId, id, "", errorText, current?.attachmentIds);
                session.setLoadingMessage(chatId, null);
              },
              onFinishReason: (reason) => {
                if (reason === "ContextLimitReached") {
                  toast.warning("Conversation reached the model's context limit; reply may be truncated");
                } else if (reason === "Length") {
                  toast.warning("Reply hit the output length limit and may be truncated");
                }
              },
            });
            if (canceled()) return;

            if (
              modelChatNamingEnabled &&
              !history.some((m) => m.sender === Roles.User) &&
              result?.finishReason === "Stop" &&
              !result.error &&
              !result.chatName
            ) {
              const titleGenResult = await useChatStore.getState().generateChatTitle(chatId, message);
              session.consumeTitleGenAbort(chatId);
              if (!titleGenResult.ok && titleGenResult.error) {
                toast.error(`Could not name the chat: ${titleGenResult.error}`);
              }
            }
          } catch (e) {
            await failReply(`Error: ${e instanceof Error ? e.message : String(e)}`);
          }
        },
        chatId,
      );
      if (accepted === null) toast.error("Model is not ready yet, please wait");
    },
    [chatId, composer, globalInstructions, runStatusBlockReason, toast, setScrollTargetId, startStream],
  );
};
