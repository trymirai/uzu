import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import { isOtherChatGenerating } from "@/features/runtime/runtime-busy";
import { ejectAndWait } from "@/features/runtime/eject-runtime-session";
import type { RuntimeSessionRef } from "@/types/session";
import { ChatRunBlockReason, type ChatRunReadyResult } from "../types";

type UiContext = {
  chatId: string;
  modelId: string | null | undefined;
};

const getUiBlockReason = (ctx: UiContext): ChatRunBlockReason | null => {
  if (!ctx.modelId) return ChatRunBlockReason.NoModel;
  if (useChatStore.getState().autoSelectSuppressed?.[ctx.chatId]) return ChatRunBlockReason.AutoSelectSuppressed;
  if (isOtherChatGenerating(useChatSessionStore.getState(), ctx.chatId)) return ChatRunBlockReason.OtherChatGenerating;
  return null;
};

const getSessionBlockReason = (target: RuntimeSessionRef): ChatRunBlockReason | null => {
  const state = useChatSessionStore.getState();

  const residentSession = useRuntimeSessionStore.getState().residentSession;
  const residentConflict = !!residentSession && residentSession.repoId !== target.repoId;

  if (state.isModelLoading) return ChatRunBlockReason.Loading;
  if (state.isEjecting || state.operationState === "ejecting") return ChatRunBlockReason.Ejecting;
  if (state.isTitleGenerating) return ChatRunBlockReason.TitleGenerating;
  if (state.isGenerating || state.operationState === "running") return ChatRunBlockReason.Running;
  if (state.operationState === "stopping") return ChatRunBlockReason.Stopping;
  if (residentConflict) return ChatRunBlockReason.ResidentConflict;
  return null;
};

export const getChatRunBlockMessage = (reason: ChatRunBlockReason): string => {
  switch (reason) {
    case ChatRunBlockReason.NoModel:
      return "Please select a model first.";
    case ChatRunBlockReason.AutoSelectSuppressed:
      return "Please select a model first.";
    case ChatRunBlockReason.OtherChatGenerating:
      return "Another chat is generating a response.";
    case ChatRunBlockReason.Running:
      return "A response is already generating.";
    case ChatRunBlockReason.Stopping:
      return "Stopping current generation, please wait";
    case ChatRunBlockReason.TitleGenerating:
      return "Generating chat title, please wait";
    case ChatRunBlockReason.Ejecting:
      return "Ejecting previous model, please wait";
    case ChatRunBlockReason.Loading:
      return "Loading model, please wait";
    case ChatRunBlockReason.EjectFailed:
      return "Failed to eject previous model";
    case ChatRunBlockReason.ResidentConflict:
      return "Model is not ready yet, please wait";
    default:
      return "Model is not ready yet, please wait";
  }
};

type PreflightOptions = {
  onEjectStart?: () => void;
};

export const prepareChatModelForRun = async (
  ctx: UiContext,
  target: RuntimeSessionRef,
  options?: PreflightOptions,
): Promise<ChatRunReadyResult> => {
  const uiReason = getUiBlockReason(ctx);
  if (uiReason) return { ok: false, reason: uiReason };

  const initialReason = getSessionBlockReason(target);
  if (initialReason === ChatRunBlockReason.ResidentConflict) {
    const state = useChatSessionStore.getState();
    const resident = useRuntimeSessionStore.getState().residentSession;
    if (!resident || !state.canEject()) return { ok: false, reason: initialReason };
    options?.onEjectStart?.();
    const ejected = await ejectAndWait(resident).catch((error: unknown) => {
      console.error("[chat] eject before run failed", { repoId: resident.repoId }, error);
      return false;
    });
    if (!ejected) return { ok: false, reason: ChatRunBlockReason.EjectFailed };
  } else if (initialReason) {
    return { ok: false, reason: initialReason };
  }

  const nextReason = getSessionBlockReason(target);
  return nextReason ? { ok: false, reason: nextReason } : { ok: true };
};
