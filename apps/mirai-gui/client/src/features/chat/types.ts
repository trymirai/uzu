export { Roles } from "@/types/chat";
export type { ChatRole, NonSystemRole } from "@/types/chat";

export enum ChatRunBlockReason {
  NoModel = "noModel",
  AutoSelectSuppressed = "autoSelectSuppressed",
  OtherChatGenerating = "otherChatGenerating",
  Loading = "loading",
  Ejecting = "ejecting",
  Running = "running",
  Stopping = "stopping",
  TitleGenerating = "titleGenerating",
  ResidentConflict = "residentConflict",
  EjectFailed = "ejectFailed",
}

export type ChatRunReadyResult = { ok: true } | { ok: false; reason: ChatRunBlockReason };
