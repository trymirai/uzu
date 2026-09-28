import type { AttachedFile } from "@/types/files";
import { SUPPORTED_FILE_TYPES, MAX_TOTAL_ATTACHMENTS_SIZE, MAX_SINGLE_FILE_SIZE } from "@/types/files";

const generateFileId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
};

const getFileExtension = (filename: string): string => {
  return filename.split(".").pop()?.toLowerCase() || "";
};

export const isValidFileType = (file: File): boolean => {
  const extension = getFileExtension(file.name);
  return SUPPORTED_FILE_TYPES.text.includes(`.${extension}`);
};

export const isValidFileSize = (file: File, currentFiles: AttachedFile[] = []): boolean => {
  if (file.size > MAX_SINGLE_FILE_SIZE) return false;
  const currentTotalSize = currentFiles.reduce((sum, f) => sum + f.size, 0);
  const newTotalSize = currentTotalSize + file.size;
  return newTotalSize <= MAX_TOTAL_ATTACHMENTS_SIZE;
};

export const processFile = async (file: File): Promise<AttachedFile> => {
  const extension = getFileExtension(file.name);
  if (!SUPPORTED_FILE_TYPES.text.includes(`.${extension}`)) {
    throw new Error("Unsupported file type");
  }
  if (file.size > MAX_SINGLE_FILE_SIZE) {
    throw new Error("File too large");
  }

  const text = await file.text();

  return {
    id: generateFileId(),
    name: file.name,
    mimeType: file.type,
    size: file.size,
    content: text,
    extension,
  };
};

export const getFileIcon = (extension: string): string => {
  const iconMap: Record<string, string> = {
    txt: "FileText",
    md: "FileText",
    json: "Code",
    csv: "Table",
    tsv: "Table",
    py: "Code",
    js: "Code",
    ts: "Code",
    tsx: "Code",
    jsx: "Code",
    html: "Code",
    css: "Code",
    xml: "Code",
    yaml: "Code",
    yml: "Code",
  };

  return iconMap[extension] || "File";
};

export { SUPPORTED_FILE_TYPES, MAX_TOTAL_ATTACHMENTS_SIZE, MAX_SINGLE_FILE_SIZE };
