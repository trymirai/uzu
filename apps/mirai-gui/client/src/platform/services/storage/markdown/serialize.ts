import { format } from "date-fns";
import type { Message, MessageVersion as StoreMessageVersion, PerfStats } from "@/types/message";
import type { ChatData } from "..";
import type { TranscriptItem } from "@/types/llm-stream";

const sanitize = (text: string): string => text.replace(/\uFFFD/g, "");

const formatTranscript = (items?: TranscriptItem[]): string[] =>
  items === undefined
    ? []
    : [`<!-- TRANSCRIPT: ${JSON.stringify(items).replace(/</g, "\\u003c").replace(/>/g, "\\u003e")} -->`];

const formatPerfToMarkdown = (perf?: PerfStats): string[] => {
  if (!perf) return [];
  const entries = [
    ["TTFT", perf.ttftSec, (v: number) => `${v.toFixed(3)}s`],
    ["TPS", perf.tps, (v: number) => `${Math.round(v)}`],
    ["Total", perf.totalSec, (v: number) => `${v.toFixed(3)}s`],
    ["TokensOut", perf.tokensOut, (v: number) => `${v}`],
  ] as const;
  const lines = entries
    .filter(([, value]) => typeof value === "number" && (value as number) > 0)
    .map(([label, value, fmt]) => `**${label}:** ${fmt(value as number)}`);
  return lines.length ? ["<!-- START_PERF -->", ...lines, "<!-- END_PERF -->"] : [];
};

const formatVersionToMarkdown = (version: StoreMessageVersion, index: number, isActive: boolean): string => {
  const cot = version.output?.text?.parsed?.chainOfThought;
  const response = version.output?.text?.parsed?.response ?? (cot ? "" : version.text);
  const errorLines = version.error ? ["<!-- START_ERROR -->", sanitize(version.error), "<!-- END_ERROR -->"] : [];
  return [
    `#### Version ${index + 1}${isActive ? " ⭐ ACTIVE" : ""}`,
    `<!-- VID: ${version.id} -->`,
    `**Time:** ${format(version.timestamp, "PPpp")}`,
    ...(version.modelName ? [`**Model:** ${version.modelName}`] : []),
    ...formatPerfToMarkdown(version.perf),
    ...(version.attachmentIds?.length ? [`**Attachments:** ${version.attachmentIds.join(",")}`] : []),
    ...errorLines,
    ...formatTranscript(version.output?.transcript),
    ...(cot ? ["<!-- START_COT -->", sanitize(cot), "<!-- END_COT -->"] : []),
    "",
    "<!-- START_CONTENT -->",
    sanitize(response || ""),
    "<!-- END_CONTENT -->",
    "",
  ].join("\n");
};

const formatMessageToMarkdown = (message: Message): string => {
  const sender = message.sender === "user" ? "👤 User" : "🤖 Assistant";
  const modelInfo = message.modelName ? ` (${message.modelName})` : "";
  const header = `## ${sender}${modelInfo} - ${format(message.timestamp, "PPpp")}\n\n`;

  if (message.versions?.length) {
    const versionsContent = message.versions
      .map((v, i) => formatVersionToMarkdown(v, i, i === message.currentVersionIndex!))
      .join("\n");
    return (
      header +
      `<!-- ID: ${message.id} -->\n### 📝 Versions (${message.versions.length})\n\n` +
      versionsContent +
      "---\n\n"
    );
  }

  const cot = message.output?.text?.parsed?.chainOfThought;
  const response = message.output?.text?.parsed?.response ?? (cot ? "" : message.text);
  const errorLines = message.error ? ["<!-- START_ERROR -->", sanitize(message.error), "<!-- END_ERROR -->"] : [];

  return (
    header +
    [
      `<!-- ID: ${message.id} -->`,
      ...formatPerfToMarkdown(message.perf),
      ...(message.attachmentIds?.length ? [`**Attachments:** ${message.attachmentIds.join(",")}`] : []),
      ...errorLines,
      ...formatTranscript(message.output?.transcript),
      ...(cot ? ["<!-- START_COT -->", sanitize(cot), "<!-- END_COT -->"] : []),
      "<!-- START_CONTENT -->",
      sanitize(response || ""),
      "<!-- END_CONTENT -->",
      "",
      "---",
      "",
    ].join("\n")
  );
};

export const serializeToMarkdown = (chat: ChatData): string =>
  [
    `# ${chat.metadata.title}`,
    "",
    `**Model:** ${chat.metadata.modelName || "Unknown"}`,
    ...(chat.metadata.modelId ? [`**ModelId:** ${chat.metadata.modelId}`] : []),
    `**Created:** ${format(chat.metadata.createdAt, "PPpp")}`,
    `**Updated:** ${format(chat.metadata.updatedAt, "PPpp")}`,
    `**Messages:** ${chat.metadata.messageCount}`,
    "",
    "---",
    "",
    ...chat.messages.map(formatMessageToMarkdown),
  ].join("\n");
