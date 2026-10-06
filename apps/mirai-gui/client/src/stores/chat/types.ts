import type { ChatState } from "../use-chat-store";

export type ChatStoreApi = {
  get: () => ChatState;
  set: (partial: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void;
};
