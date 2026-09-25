import type { ChatState } from "../useChatStore";

export type ChatStoreApi = {
  get: () => ChatState;
  set: (partial: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void;
};
