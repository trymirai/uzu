import { useEffect } from "react";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { runtimeSessionEjectReasons } from "@/types/session";

type UseEjectDrivenModelResetParams = {
  chatId: string;
  selectedModelId: string | null;
  pendingSwitch: boolean;
};

// After a user-initiated eject the chat shows "Select model" instead of
// snapping to another resident model; waits for idle so it does not fire
// mid-eject or mid-load.
export const useEjectDrivenModelReset = ({
  chatId,
  selectedModelId,
  pendingSwitch,
}: UseEjectDrivenModelResetParams): void => {
  const lastEjected = useChatSessionStore((s) => s.lastEjected);
  const lastEjectedReason = useChatSessionStore((s) => s.lastEjectedReason);
  const setLastEjected = useChatSessionStore((s) => s.setLastEjected);
  const operationState = useChatSessionStore((s) => s.operationState);
  const isGenerating = useChatSessionStore((s) => s.isGenerating);
  const isModelLoading = useChatSessionStore((s) => s.isModelLoading);
  const isEjecting = useChatSessionStore((s) => s.isEjecting);
  const isTitleGenerating = useChatSessionStore((s) => s.isTitleGenerating);
  const suppressAutoSelect = useChatStore((s) => s.suppressAutoSelect);
  const setChatModel = useChatStore((s) => s.setChatModel);

  useEffect(() => {
    if (!lastEjected) return;
    if (pendingSwitch) return;
    const idle = operationState === "idle" && !isGenerating && !isModelLoading && !isEjecting && !isTitleGenerating;
    if (!idle) return;
    if (lastEjectedReason !== runtimeSessionEjectReasons.user) return;

    const matchesEjected = selectedModelId && selectedModelId === lastEjected.repoId;

    if (matchesEjected) {
      suppressAutoSelect(chatId, true);
      setChatModel(chatId, "", "");
      setLastEjected(null);
    }
  }, [
    lastEjected,
    lastEjectedReason,
    setLastEjected,
    selectedModelId,
    suppressAutoSelect,
    setChatModel,
    chatId,
    operationState,
    isGenerating,
    isModelLoading,
    isEjecting,
    isTitleGenerating,
    pendingSwitch,
  ]);
};
