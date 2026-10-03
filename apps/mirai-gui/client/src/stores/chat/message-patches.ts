import type { Message, MessageVersion, ParsedOutput } from "@/types/message";
import type { OutputShape, ParsedPatch } from "@/types/llm-stream";

export function withParsedOutput(output: OutputShape | undefined, patch: ParsedPatch): OutputShape {
  return {
    ...(output || {}),
    text: {
      ...(output?.text || {}),
      parsed: {
        ...(output?.text?.parsed || {}),
        ...(patch.chainOfThought !== undefined ? { chainOfThought: patch.chainOfThought } : {}),
        ...(patch.response !== undefined ? { response: patch.response } : {}),
      },
    },
  };
}

const mergeParsedIntoOutput = (
  output: Message["output"] | undefined,
  parsed?: ParsedOutput,
): Message["output"] | undefined => (parsed ? withParsedOutput(output, parsed) : output);

export const computeErrorPatch = (
  message: Message,
  text: string,
  error: string,
  attachmentIds?: string[],
): Partial<Message> => {
  const hasVersions = Array.isArray(message.versions) && (message.versions.length || 0) > 0;
  if (hasVersions) {
    const versionsArr = message.versions || [];
    const idx = typeof message.currentVersionIndex === "number" ? message.currentVersionIndex : versionsArr.length - 1;
    const versions: MessageVersion[] = versionsArr.map((v, i): MessageVersion =>
      i === idx ? { ...v, text, error, ...(attachmentIds ? { attachmentIds } : {}) } : v,
    );
    return { text, error, versions, ...(attachmentIds ? { attachmentIds } : {}) };
  }
  return { text, error, ...(attachmentIds ? { attachmentIds } : {}) };
};

export const computeFinalizedUpdates = (message: Message, text: string, parsed?: ParsedOutput): Partial<Message> => {
  const hasVersions = Array.isArray(message.versions) && (message.versions.length || 0) > 0;
  if (hasVersions) {
    const versionsArr = message.versions || [];
    const idx = typeof message.currentVersionIndex === "number" ? message.currentVersionIndex : versionsArr.length - 1;
    const versions: MessageVersion[] = versionsArr.map((v, i): MessageVersion =>
      i === idx
        ? {
            ...v,
            text,
            error: undefined,
            output: mergeParsedIntoOutput(v.output, parsed),
          }
        : v,
    );
    return { text, error: undefined, versions };
  }
  const output = mergeParsedIntoOutput(message.output, parsed);
  return { text, error: undefined, ...(output ? { output } : {}) };
};
