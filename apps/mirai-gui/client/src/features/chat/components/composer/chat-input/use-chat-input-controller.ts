import type { KeyboardEvent as ReactKeyboardEvent, MutableRefObject } from "react";
import { useCallback, useMemo, useRef } from "react";
import { isMacPlatform } from "@/utils/platform";
import type { ChatInputFile, ChatInputSendPayload } from "./types";

const EMPTY_FILES: ChatInputFile[] = [];

export type UseChatInputControllerArgs = {
  value: string;
  files?: ChatInputFile[];
  onSend?: (payload: ChatInputSendPayload) => void;
  canSend?: (payload: ChatInputSendPayload) => boolean;
  onBlockedSend?: (payload: ChatInputSendPayload) => void;
};

export type UseChatInputControllerResult = {
  textareaRef: MutableRefObject<HTMLTextAreaElement | null>;
  sendDisabled: boolean;
  hasContent: boolean;
  handleSubmit: () => void;
  handleKeyDown: (e: ReactKeyboardEvent<HTMLTextAreaElement>) => void;
};

export function useChatInputController({
  value,
  files,
  onSend,
  canSend,
  onBlockedSend,
}: UseChatInputControllerArgs): UseChatInputControllerResult {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const isMac = isMacPlatform();

  const effectiveFiles = files ?? EMPTY_FILES;
  const trimmedText = value.trim();
  const hasContent = trimmedText.length > 0 || effectiveFiles.length > 0;

  const payload = useMemo<ChatInputSendPayload>(
    () => ({ text: trimmedText, files: effectiveFiles }),
    [effectiveFiles, trimmedText],
  );

  const sendDisabled = !hasContent || !onSend;

  const focusTextarea = useCallback(() => {
    textareaRef.current?.focus();
  }, []);

  const handleSubmit = useCallback(() => {
    if (sendDisabled) return;
    if (!onSend) return;
    if (canSend && !canSend(payload)) {
      onBlockedSend?.(payload);
      return;
    }

    onSend(payload);
    focusTextarea();
  }, [canSend, focusTextarea, onBlockedSend, onSend, payload, sendDisabled]);

  const handleKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      const metaOrCtrl = isMac ? e.metaKey : e.ctrlKey;

      if (e.key !== "Enter") return;
      // In WebKit the Enter that confirms an IME choice arrives with isComposing
      // already false and keyCode 229.
      if (e.nativeEvent.isComposing || e.nativeEvent.keyCode === 229) return;

      if (metaOrCtrl) {
        if (!hasContent || !onSend) return;
        e.preventDefault();
        handleSubmit();
        return;
      }

      if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
      if (!hasContent || !onSend) return;

      e.preventDefault();
      handleSubmit();
    },
    [handleSubmit, hasContent, isMac, onSend],
  );

  return {
    textareaRef,
    sendDisabled,
    hasContent,
    handleSubmit,
    handleKeyDown,
  };
}
