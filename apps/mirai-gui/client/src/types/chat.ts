export type ChatRole = "system" | "user" | "assistant";

export const Roles = {
  System: "system",
  User: "user",
  Assistant: "assistant",
} as const;

export type NonSystemRole = Exclude<ChatRole, "system">;
