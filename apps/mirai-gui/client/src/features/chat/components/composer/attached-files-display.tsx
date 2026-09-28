import { twMerge } from "tailwind-merge";
import type { AttachedFile } from "@/types/files";
import { FileIcon } from "./file-icon";

type FileType = "document" | "code" | "table" | "other";

const DOCUMENT_EXTENSIONS = ["txt", "md"];
const CODE_EXTENSIONS = ["json", "py", "js", "ts", "tsx", "jsx", "html", "css", "xml", "yaml", "yml"];
const TABLE_EXTENSIONS = ["csv", "tsv"];

const LABEL_COLOR: Record<FileType, string> = {
  document: "text-blue",
  code: "text-progress",
  table: "text-green-500",
  other: "text-label-muted dark:text-label-muted-dark",
};

const ICON_BACKGROUND: Record<FileType, string> = {
  document: "bg-blue/[0.12] text-blue border-blue/[0.12]",
  code: "bg-progress/[0.12] text-progress border-progress/[0.12]",
  table: "bg-green-500/[0.12] text-green-500 border-green-500/[0.12]",
  other: "bg-bg-hover dark:bg-bg-hover-dark border-button-border dark:border-button-border-dark",
};

const getFileType = (extension: string): FileType => {
  if (DOCUMENT_EXTENSIONS.includes(extension)) return "document";
  if (CODE_EXTENSIONS.includes(extension)) return "code";
  if (TABLE_EXTENSIONS.includes(extension)) return "table";
  return "other";
};

const getFileNameWithoutExtension = (filename: string): string => {
  const lastDotIndex = filename.lastIndexOf(".");
  return lastDotIndex > 0 ? filename.substring(0, lastDotIndex) : filename;
};

function FileCard({ file }: { file: AttachedFile }) {
  const fileType = getFileType(file.extension);
  return (
    <div className="group relative flex items-center gap-2 px-2 py-[6px] rounded-[8px] border max-w-[200px] border-cell-border dark:border-cell-border-dark">
      <div
        className={twMerge(
          "w-10 h-10 border rounded-[5px] flex items-center justify-center flex-shrink-0",
          ICON_BACKGROUND[fileType],
        )}
      >
        <FileIcon extension={file.extension} className="w-4 h-4" />
      </div>

      <div className="flex flex-col min-w-0 flex-1 gap-[2px]">
        <span className="text-[13px] font-[350] leading-[150%] text-label-title dark:text-label-title-dark truncate">
          {getFileNameWithoutExtension(file.name)}
        </span>
        <span className={twMerge("text-xs leading-[130%]", LABEL_COLOR[fileType])}>{file.extension}</span>
      </div>
    </div>
  );
}

export function AttachedFilesDisplay({ files }: { files: AttachedFile[] }) {
  if (files.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {files.map((file) => (
        <FileCard key={file.id} file={file} />
      ))}
    </div>
  );
}
