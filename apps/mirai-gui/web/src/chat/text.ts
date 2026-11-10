import type { LlmRunParams, ParsedPatch } from "@/types/llm-stream";

export type HistoryMessage = LlmRunParams["messages"][number];

// Engine output only: U+FFFD shows up as "�" when the decoder splits a multi-byte
// character across tokens. Joiners and variation selectors hold composite emoji together.
// eslint-disable-next-line no-control-regex
const UNSAFE = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\uFFFC\uFFFD\uFEFF]/g;

export const sanitize = (value: string): string => value.replace(UNSAFE, "");

export const joinTurns = (first: string, second: string): string =>
  !first ? second : !second ? first : `${first}\n\n${second}`;

export function parsedFrom(text: string, reasoning: string): ParsedPatch | undefined {
  const chain = reasoning.trim();
  const response = text.trim();
  if (!chain && !response) return undefined;
  return {
    ...(chain ? { chainOfThought: sanitize(chain) } : {}),
    ...(response ? { response: sanitize(response) } : {}),
  };
}

// Match the final text/parsed fields sent to the client. The resident session
// keeps the original reasoning bytes and internal tool messages.
export function assistantHistoryMessage(
  text: string,
  reasoning: string,
  promotedReasoning: boolean,
): HistoryMessage | undefined {
  if (promotedReasoning || !text.trim()) return undefined;
  const reasoningContent = sanitize(reasoning.trim());
  return { role: "assistant", content: sanitize(text), ...(reasoningContent ? { reasoningContent } : {}) };
}

export const sameHistoryMessage = (a: HistoryMessage, b: HistoryMessage): boolean =>
  a.role === b.role &&
  a.content === b.content &&
  (a.reasoningContent || undefined) === (b.reasoningContent || undefined);
