import { useCallback } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import type { AttachedFile } from "@/types/files";
import { Roles } from "../types";
import { attachmentStorage } from "../services/attachment-storage";
import { ChatRunBlockReason } from "../types";
import { getChatRunBlockMessage, prepareChatModelForRun } from "../services/chat-run-preflight";
import { buildChatRunInput } from "../services/build-chat-run-input";
import type { ChatComposerState } from "./use-chat-composer-state";
import type { StartStreamOptions } from "./use-llm-stream";
import type { ToastApi } from "@/components/ui/toast/use-toast";

const isTransientTitleGenError = (error: string): boolean => /Network error:/i.test(error);

type UseSendChatMessageParams = {
  chatId: string;
  composer: ChatComposerState;
  globalInstructions: string | null | undefined;
  runStatusBlockReason: ChatRunBlockReason | null;
  toast: ToastApi;
  setScrollTargetId: (id: string | null) => void;
  startStream: (options: StartStreamOptions) => Promise<void | null>;
};

export const useSendChatMessage = (params: UseSendChatMessageParams) => {
  const { chatId, composer, globalInstructions, runStatusBlockReason, toast, setScrollTargetId, startStream } = params;

  return useCallback(
    async (message: string) => {
      const session = useChatSessionStore.getState();
      useChatStore.getState().setCurrentChatId(chatId);

      const storeState = useChatStore.getState();
      const modelId = storeState.chatModels[chatId]?.modelId;
      if (!modelId) throw new Error("Model is not selected");

      // Captured before any await: the user may open another chat while this
      // one waits for an eject or a write, and the store then holds that chat.
      const history = storeState.messages;

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

      const fileIds = composer.attachedFiles.map((file: AttachedFile) => {
        attachmentStorage.saveFile(file);
        return file.id;
      });

      const userMessage = useChatStore.getState().addMessageTo(chatId, {
        text: message,
        sender: Roles.User,
        attachmentIds: fileIds.length > 0 ? fileIds : undefined,
      });
      const userId = userMessage.id;
      const filesToSend = [...composer.attachedFiles];
      composer.clear();

      setScrollTargetId(userId);

      try {
        await useChatStore.getState().persistMessage(chatId, userId, userMessage);

        const { messages: messagesForRun } = buildChatRunInput({
          history,
          prompt: message,
          attachments: filesToSend,
          globalInstructions,
        });

        // Placeholder must exist before title-gen so Stop during title-gen has a
        // message to settle instead of leaving a dangling user turn.
        const freshStore = useChatStore.getState();
        const placeholder = freshStore.addMessageTo(chatId, {
          text: "",
          sender: Roles.Assistant,
          modelId: modelId || undefined,
          modelName: freshStore.chatModels[chatId]?.modelName || undefined,
          perf: {},
          attachmentIds: fileIds.length > 0 ? fileIds : undefined,
        });
        const assistantMessageId = placeholder.id;
        session.setLoadingMessage(chatId, assistantMessageId);
        await useChatStore.getState().persistMessage(chatId, assistantMessageId, placeholder);

        const titleGenResult = await useChatStore.getState().generateChatTitle(chatId, message);

        if (useChatSessionStore.getState().consumeTitleGenAbort(chatId)) {
          // The placeholder is already on disk; left there it would come back
          // as a blank bubble after a reload.
          session.setCanceledMessage(chatId, null);
          session.setLoadingMessage(chatId, null);
          await useChatStore.getState().discardMessage(chatId, assistantMessageId);
          return;
        }

        if (!titleGenResult.ok && titleGenResult.error && !isTransientTitleGenError(titleGenResult.error)) {
          const errText = `Error: ${titleGenResult.error}`;
          useChatStore.getState().updateMessage(assistantMessageId, {
            text: "",
            error: errText,
            attachmentIds: fileIds.length > 0 ? fileIds : undefined,
          });
          await useChatStore
            .getState()
            .persistMessageError(chatId, assistantMessageId, "", errText, fileIds.length > 0 ? fileIds : undefined);
          session.setLoadingMessage(chatId, null);
          return;
        }

        void startStream({
          repoId: modelId,
          messages: messagesForRun,
          messageId: assistantMessageId,
          chatId,
          updateText: (id: string, updatedText: string) => {
            useChatStore.getState().updateMessage(id, {
              text: updatedText,
              attachmentIds: fileIds.length > 0 ? fileIds : undefined,
            });
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
      } catch (e) {
        session.setLoadingMessage(chatId, null);
        const s = useChatStore.getState();
        const errText = `Error: ${e instanceof Error ? e.message : String(e)}`;
        const errMsg = s.addMessageTo(chatId, {
          text: "",
          error: errText,
          sender: Roles.Assistant,
          modelId: s.chatModels[chatId]?.modelId || undefined,
          modelName: s.chatModels[chatId]?.modelName || undefined,
          attachmentIds: fileIds.length > 0 ? fileIds : undefined,
        });
        await useChatStore
          .getState()
          .persistMessage(chatId, errMsg.id, errMsg)
          .catch((err) => {
            console.error("[storage] failed to persist error message", err);
          });
      }
    },
    [chatId, composer, globalInstructions, runStatusBlockReason, toast, setScrollTargetId, startStream],
  );
};
