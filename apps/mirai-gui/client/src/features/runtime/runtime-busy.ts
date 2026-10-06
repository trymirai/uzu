import { useChatSessionStore } from "@/stores/use-chat-session-store";

type BusyFields = {
  isGenerating: boolean;
  isTitleGenerating: boolean;
  operationState: string;
};

type GenerationFields = {
  isGenerating: boolean;
  activeGeneratingChatId: string | null;
};

// operationState is set as soon as a run operation starts, before isGenerating:
// the previous run is released in between.
const selectRuntimeBusy = (state: BusyFields): boolean =>
  state.isGenerating ||
  state.isTitleGenerating ||
  state.operationState === "running" ||
  state.operationState === "stopping";

export const isRuntimeBusy = (): boolean => selectRuntimeBusy(useChatSessionStore.getState());

export const useRuntimeBusy = (): boolean => useChatSessionStore(selectRuntimeBusy);

export const isChatGenerating = (state: GenerationFields, chatId: string): boolean =>
  state.isGenerating && state.activeGeneratingChatId === chatId;

export const isOtherChatGenerating = (state: GenerationFields, chatId: string): boolean =>
  state.isGenerating && !!state.activeGeneratingChatId && state.activeGeneratingChatId !== chatId;
