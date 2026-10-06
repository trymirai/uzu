import JSZip from "jszip";
import { listChatFiles, loadChatFile } from "./chat-files";
import { stripHtmlComments, wrapCotInDetails } from "./markdown/parse";

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
      zip.file(file, stripHtmlComments(wrapCotInDetails(normalized)));
    } catch {
      continue;
    }
  }
  return zip.generateAsync({ type: "uint8array", compression: "DEFLATE" });
};
