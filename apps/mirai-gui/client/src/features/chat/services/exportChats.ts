import { format } from "date-fns";
import { getPlatform } from "@/platform/platformSingleton";
import { platformInfo } from "@/platform/platformInfo";

function triggerBrowserDownload(data: Uint8Array, filename: string): void {
  const blob = new Blob([data as unknown as BlobPart], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

export async function exportAllChatsZip(): Promise<boolean> {
  const { storage, dialogs } = getPlatform();
  const data = await storage.exportAllChatsZip();
  if (!data) return false;

  const humanDate = format(Date.now(), "dd.MM.yyyy HH.mm");
  const defaultName = `mirai-chats ${humanDate}.zip`;

  if (!platformInfo.features.fileSystem) {
    triggerBrowserDownload(data, defaultName);
    return true;
  }

  const targetPath = await dialogs.showSaveDialog({
    title: "Export chats",
    defaultPath: defaultName,
    filters: [{ name: "ZIP archive", extensions: ["zip"] }],
  });
  if (!targetPath) return false;
  return !!(await storage.saveBinaryFile(targetPath, data));
}
