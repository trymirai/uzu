import type { KeyboardEvent as ReactKeyboardEvent, MutableRefObject } from "react";
import { useCallback, useMemo, useRef, useState } from "react";
import { isMacPlatform } from "@/utils/platform";
import type { ChatInputFile, ChatInputSendPayload } from "./types";

const EMPTY_FILES: ChatInputFile[] = [];

export type UseChatInputControllerArgs = {
  value: string;
  onChange: (value: string) => void;
  files?: ChatInputFile[];
  onSend?: (payload: ChatInputSendPayload) => void | Promise<void>;
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
  onChange,
  files,
  onSend,
  canSend,
  onBlockedSend,
}: UseChatInputControllerArgs): UseChatInputControllerResult {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [internalBusy, setInternalBusy] = useState(false);
  const isMac = isMacPlatform();

  const effectiveFiles = files ?? EMPTY_FILES;
  const trimmedText = value.trim();
  const hasContent = trimmedText.length > 0 || effectiveFiles.length > 0;

  const payload = useMemo<ChatInputSendPayload>(
    () => ({ text: trimmedText, files: effectiveFiles }),
    [effectiveFiles, trimmedText],
  );

  const sendDisabled = internalBusy || !hasContent || typeof onSend !== "function";

  const focusTextarea = useCallback(() => {
    textareaRef.current?.focus();
  }, []);

  const clearTextarea = useCallback(() => {
    onChange("");
  }, [onChange]);

  const handleSubmit = useCallback(() => {
    if (sendDisabled) return;
    if (typeof onSend !== "function") return;
    if (typeof canSend === "function" && !canSend(payload)) {
      onBlockedSend?.(payload);
      return;
    }

    setInternalBusy(true);
    Promise.resolve(onSend(payload))
      .then(() => {
        clearTextarea();
        focusTextarea();
      })
      .finally(() => {
        setInternalBusy(false);
      });
  }, [canSend, clearTextarea, focusTextarea, onBlockedSend, onSend, payload, sendDisabled]);

  const handleKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      const metaOrCtrl = isMac ? e.metaKey : e.ctrlKey;

      if (e.key !== "Enter") return;

      if (metaOrCtrl) {
        if (!hasContent || typeof onSend !== "function") return;
        e.preventDefault();
        handleSubmit();
        return;
      }

      if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
      if (!hasContent || typeof onSend !== "function") return;

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
