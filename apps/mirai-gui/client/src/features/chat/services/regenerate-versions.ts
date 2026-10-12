import type { Message, MessageVersion } from "@/types/message";
import { v4 as uuidv4 } from "uuid";

// If the target has no versions yet, freeze its body as version 0 so
// perf/stats/output/error from the prior turn are preserved.
export const projectAssistantVersion = (
  target: Message,
  modelId: string,
  modelName: string,
  attachmentIds?: string[],
): MessageVersion[] => {
  const newVersion: MessageVersion = {
    id: uuidv4(),
    text: "",
    modelId,
    modelName,
    timestamp: Date.now(),
    perf: {},
    attachmentIds,
  };

  if (Array.isArray(target.versions) && target.versions.length > 0) {
    return [...target.versions, newVersion];
  }

  const frozenOriginal: MessageVersion = {
    id: uuidv4(),
    text: target.text,
    modelId: target.modelId || "",
    modelName: target.modelName || "",
    timestamp: target.timestamp,
    attachmentIds: target.attachmentIds,
    ...(target.perf ? { perf: target.perf } : {}),
    ...(target.stats ? { stats: target.stats } : {}),
    ...(target.error ? { error: target.error } : {}),
    ...(target.output ? { output: target.output } : {}),
  };

  return [frozenOriginal, newVersion];
};

export const patchActiveVersion = (
  versions: MessageVersion[] | undefined,
  patch: Partial<MessageVersion>,
): MessageVersion[] => {
  const active = versions?.at(-1);
  if (!versions || !active) return versions ?? [];
  return [...versions.slice(0, -1), { ...active, ...patch }];
};
