import type { Message, PerfStats, MessageVersion as StoreMessageVersion } from "@/types/message";
import { parse } from "date-fns";
import type { ChatMetadata } from "..";

const extractContent = (text: string): string => {
  const match = text.match(/<!-- START_CONTENT -->\n([\s\S]*?)\n<!-- END_CONTENT -->/);
  return match?.[1] ?? "";
};

const extractChainOfThought = (text: string): string | undefined => {
  const match = text.match(/<!-- START_COT -->\n([\s\S]*?)\n<!-- END_COT -->/m);
  return match?.[1] || undefined;
};

const extractError = (text: string): string | undefined => {
  const match = text.match(/<!-- START_ERROR -->\n([\s\S]*?)\n<!-- END_ERROR -->/m);
  return match?.[1] || undefined;
};

const parseDate = (s: string | undefined): number => {
  if (!s) return Date.now();
  const t = parse(s, "PPpp", new Date()).getTime();
  return Number.isNaN(t) ? Date.now() : t;
};

const parsePerf = (block: string): PerfStats | undefined => {
  const blockMatch = block.match(/<!-- START_PERF -->\n([\s\S]*?)\n<!-- END_PERF -->/m);
  const scope = blockMatch?.[1] ?? block;
  const toNum = (m: RegExpMatchArray | null) => (m?.[1] ? Number(m[1]) : undefined);
  const perf: PerfStats = {
    ttftSec:
      toNum(scope.match(/\*\*TTFT:\*\*\s*([0-9]+(?:\.[0-9]+)?)s/m)) ??
      toNum(scope.match(/\*\*TTFB:\*\*\s*([0-9]+(?:\.[0-9]+)?)s/m)),
    tps: toNum(scope.match(/\*\*TPS:\*\*\s*([0-9]+(?:\.[0-9]+)?)/m)),
    totalSec: toNum(scope.match(/\*\*Total:\*\*\s*([0-9]+(?:\.[0-9]+)?)s/m)),
    tokensOut: toNum(scope.match(/\*\*TokensOut:\*\*\s*([0-9]+)/m)),
  };
  return Object.values(perf).some((v) => typeof v === "number") ? perf : undefined;
};

const parseVersions = (block: string): { versions: StoreMessageVersion[]; currentVersionIndex: number } => {
  const headerRegex = /^#### Version \d+.*$/gm;
  const headers: Array<{ start: number; header: string }> = [];
  let match: RegExpExecArray | null;
  while ((match = headerRegex.exec(block)) !== null) {
    headers.push({ start: match.index, header: match[0] });
  }

  let currentVersionIndex = 0;
  const versions = headers
    .map((h, idx) => {
      const content = block.slice(h.start, headers[idx + 1]?.start ?? block.length);
      const isActive = h.header.includes("⭐ ACTIVE");
      if (isActive) currentVersionIndex = idx;

      const timeMatch = content.match(/\*\*Time:\*\* (.+)$/m);
      if (!timeMatch) return null;

      const idMatch = content.match(/<!-- VID:\s*([^>]+?)\s*-->/m);
      const modelMatch = content.match(/\*\*Model:\*\* (.+)$/m);
      const attachmentIds = content.match(/\*\*Attachments:\*\* (.+)$/m)?.[1]?.split(",");
      const versionText = extractContent(content);
      const cot = extractChainOfThought(content);
      const versionError = extractError(content);
      const versionPerf = parsePerf(content);
      const outputParsed =
        cot || versionText
          ? {
              text: {
                parsed: { ...(cot ? { chainOfThought: cot } : {}), ...(versionText ? { response: versionText } : {}) },
              },
            }
          : undefined;

      return {
        id: idMatch?.[1] || `${Date.now()}-${idx}`,
        modelId: "",
        modelName: modelMatch?.[1] || "",
        timestamp: parseDate(timeMatch[1]),
        text: versionText,
        ...(outputParsed ? { output: outputParsed } : {}),
        ...(versionPerf ? { perf: versionPerf } : {}),
        ...(attachmentIds ? { attachmentIds } : {}),
        ...(versionError ? { error: versionError } : {}),
      };
    })
    .filter((v): v is NonNullable<typeof v> => v !== null) as StoreMessageVersion[];

  return {
    versions,
    currentVersionIndex: Math.min(currentVersionIndex, Math.max(0, versions.length - 1)),
  };
};

export const parseMessage = (block: string): Message | null => {
  const headerMatch = block.match(/^## (👤|🤖) (.+?) - (.+)$/m);
  if (!headerMatch) return null;

  const [, marker, who, when] = headerMatch;
  const sender = marker === "👤" ? "user" : "assistant";
  const timestamp = parseDate(when);
  const modelName = who?.match(/^(.+?) \((.+)\)$/)?.[2];
  const idMatch = block.match(/<!-- ID:\s*([^>]+?)\s*-->/m);
  const attachmentIds = block.match(/\*\*Attachments:\*\* (.+)$/m)?.[1]?.split(",");
  const id = idMatch?.[1] || `${Date.now()}-${Math.random()}`;

  if (block.includes("### 📝 Versions")) {
    const { versions, currentVersionIndex } = parseVersions(block);
    return {
      id,
      text: versions[currentVersionIndex]?.text || "",
      sender: sender as Message["sender"],
      modelName,
      timestamp,
      versions,
      currentVersionIndex,
      ...(attachmentIds ? { attachmentIds } : {}),
    };
  }

  const cot = extractChainOfThought(block);
  const resp = extractContent(block);
  const errorText = extractError(block);
  const perf = parsePerf(block);
  const outputParsed =
    cot || resp
      ? { text: { parsed: { ...(cot ? { chainOfThought: cot } : {}), ...(resp ? { response: resp } : {}) } } }
      : undefined;

  return {
    id,
    text: resp,
    sender: sender as Message["sender"],
    modelName,
    timestamp,
    ...(outputParsed ? { output: outputParsed } : {}),
    ...(perf ? { perf } : {}),
    ...(attachmentIds ? { attachmentIds } : {}),
    ...(errorText ? { error: errorText } : {}),
  };
};

export const extractMessageBlocks = (markdown: string): string[] => {
  const blocks: string[] = [];
  const headerRegex = /^## (👤|🤖) .+ - .+$/gm;
  let match: RegExpExecArray | null;
  while ((match = headerRegex.exec(markdown)) !== null) {
    const start = match.index;
    const endContentIdx = markdown.indexOf("<!-- END_CONTENT -->", start);
    if (endContentIdx === -1) continue;
    const delimRegex = /\n---\r?\n/gm;
    delimRegex.lastIndex = endContentIdx;
    const dMatch = delimRegex.exec(markdown);
    blocks.push(markdown.slice(start, dMatch ? dMatch.index + dMatch[0].length : markdown.length));
  }
  return blocks;
};

export const parseMetadata = (markdown: string, chatId: string, messagesCount: number): ChatMetadata => {
  return {
    id: chatId,
    title: markdown.match(/^# (.+)$/m)?.[1] ?? "Untitled",
    modelId: markdown.match(/\*\*ModelId:\*\* (.+)$/m)?.[1],
    modelName: markdown.match(/\*\*Model:\*\* (.+)$/m)?.[1],
    createdAt: parseDate(markdown.match(/\*\*Created:\*\* (.+)$/m)?.[1]),
    updatedAt: parseDate(markdown.match(/\*\*Updated:\*\* (.+)$/m)?.[1]),
    messageCount: messagesCount,
  };
};

export function wrapCotInDetails(markdown: string): string {
  return markdown.replace(/(<!-- START_COT -->\n)([\s\S]*?)(\n<!-- END_COT -->)/g, (_match, start, body, end) => {
    const trimmedBody = body.replace(/^\n+|\n+$/g, "");
    return [
      "<details>",
      "<summary>Thinking</summary>",
      "",
      start.trimEnd(),
      trimmedBody,
      end.trimStart(),
      "",
      "</details>",
    ].join("\n");
  });
}

export const stripHtmlComments = (markdown: string): string => markdown.replace(/<!--[\s\S]*?-->/g, () => "");
