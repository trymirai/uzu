export const navigationRequestTypes = {
  openChatForModel: "open-chat-for-model",
  newChat: "new-chat",
  openPreferences: "open-preferences",
} as const;

export type NavigationRequest =
  | { type: typeof navigationRequestTypes.newChat }
  | { type: typeof navigationRequestTypes.openPreferences };
