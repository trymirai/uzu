import JSZip from "jszip";
import { listChatFiles, loadChatFile } from "./chat-files";
import { stripHtmlComments, wrapCotInDetails } from "./markdown/parse";
import { isChartSpec } from "@/types/chart";
import { chartToMarkdown } from "@/utils/chart-data";
import { transcriptMetadata } from "./markdown/transcript-metadata";

const includeChartTables = (markdown: string): string => {
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
      zip.file(file, stripHtmlComments(wrapCotInDetails(includeChartTables(normalized))));
    } catch {
      continue;
    }
  }
  return zip.generateAsync({ type: "uint8array", compression: "DEFLATE" });
};
