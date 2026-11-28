import JSZip from "jszip";
import { listChatFiles, loadChatFile } from "./chat-files";
import { stripHtmlComments, wrapCotInDetails } from "./markdown/parse";
import { isChartSpec } from "@/types/chart";
import { isWebPageItem, isWebSearchItem } from "@/types/web-search";
import { chartToMarkdown } from "@/utils/chart-data";
import { collectSources } from "@/utils/web-sources";
import { transcriptMetadata } from "./markdown/transcript-metadata";

// A title is a page's own text: it must not end the link early or pass for a comment the export drops.
const sourcesToMarkdown = (items: unknown[]): string => {
  const sources = collectSources(items.filter((item) => isWebSearchItem(item) || isWebPageItem(item)));
  const lines = sources.map(({ title, url }) => {
    const text = (title || url).replace(/[\\[\]]/g, "\\$&").replace(/</g, "&lt;");
    return `- [${text}](${url.replace(/\(/g, "%28").replace(/\)/g, "%29")})`;
  });
  return lines.length > 0 ? `Sources:\n\n${lines.join("\n")}` : "";
};

const ANSWER_END = "\n<!-- END_CONTENT -->";
const NEXT_MESSAGE = /^(?:## (?:👤|🤖) |#### Version \d)/m;

const answerEnd = (markdown: string, from: number): number | undefined => {
  const end = markdown.indexOf(ANSWER_END, from);
  const next = markdown.slice(from).search(NEXT_MESSAGE);
  return end !== -1 && (next === -1 || end < from + next) ? end : undefined;
};

const includeTranscriptData = (markdown: string): string => {
  let offset = 0;
  let result = "";
  for (const { start, end, json } of transcriptMetadata(markdown)) {
    result += markdown.slice(offset, start);
    offset = end;
    try {
      const items: unknown = JSON.parse(json);
      if (!Array.isArray(items)) continue;
      const charts = items
        .filter((item) => item?.type === "chart")
        .map((item) =>
          isChartSpec(item.chart) ? chartToMarkdown(item.chart) : "_This chart could not be exported: invalid data._",
        )
        .join("\n\n");
      if (charts) result += `\n${charts}\n`;
      const sources = sourcesToMarkdown(items);
      const after = sources ? (answerEnd(markdown, end) ?? end) : end;
      if (sources) {
        result += `${markdown.slice(offset, after)}\n\n${sources}\n`;
        offset = after;
      }
    } catch {
      // Invalid metadata was previously stripped with the other HTML comments.
    }
  }
  return result + markdown.slice(offset);
};

export const buildChatsZip = async (): Promise<Uint8Array | null> => {
  const zip = new JSZip();
  const files = await listChatFiles();
  const mdFiles = files.filter((f) => f.endsWith(".md"));
  if (mdFiles.length === 0) return null;
  for (const file of mdFiles) {
    try {
      const content = await loadChatFile(file);
      if (!content) continue;
      const normalized = content.replace(/\*\*TTFB:\*\*/g, "**TTFT:**");
      zip.file(file, stripHtmlComments(wrapCotInDetails(includeTranscriptData(normalized))));
    } catch {
      continue;
    }
  }
  return zip.generateAsync({ type: "uint8array", compression: "DEFLATE" });
};
