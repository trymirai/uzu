import { useChatSessionStore } from "@/stores/useChatSessionStore";
import { useChatStore } from "@/stores/useChatStore";
import { isOtherChatGenerating } from "@/features/runtime/runtimeBusy";
import { ChatRunBlockReason } from "../types";

type UseChatRunStatusParams = {
  chatId: string;
  modelId: string | null | undefined;
};

type UseChatRunStatusResult = {
  blockReason: ChatRunBlockReason | null;
  canRun: boolean;
};

export const useChatRunStatus = ({ chatId, modelId }: UseChatRunStatusParams): UseChatRunStatusResult => {
  const autoSelectSuppressed = useChatStore((s) => !!s.autoSelectSuppressed?.[chatId]);
  const otherChatGenerating = useChatSessionStore((s) => isOtherChatGenerating(s, chatId));

  let blockReason: ChatRunBlockReason | null = null;
  if (!modelId) blockReason = ChatRunBlockReason.NoModel;
  else if (autoSelectSuppressed) blockReason = ChatRunBlockReason.AutoSelectSuppressed;
  else if (otherChatGenerating) blockReason = ChatRunBlockReason.OtherChatGenerating;

  return { blockReason, canRun: blockReason === null };
};
