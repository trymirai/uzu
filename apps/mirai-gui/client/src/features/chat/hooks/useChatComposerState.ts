import { useCallback, useState } from "react";
import type { AttachedFile } from "@/types/files";

export type ChatComposerState = {
  attachedFiles: AttachedFile[];
  attachFile: (file: AttachedFile) => void;
  removeFile: (fileId: string) => void;
  clear: () => void;
};

export const useChatComposerState = (): ChatComposerState => {
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);

  const attachFile = useCallback((file: AttachedFile) => {
    setAttachedFiles((prev) => [...prev, file]);
  }, []);

  const removeFile = useCallback((fileId: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  const clear = useCallback(() => {
    setAttachedFiles([]);
  }, []);

  return { attachedFiles, attachFile, removeFile, clear };
};
