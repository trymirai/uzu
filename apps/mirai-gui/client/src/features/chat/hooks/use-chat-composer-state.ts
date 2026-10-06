import { useCallback, useState } from "react";
import type { AttachedFile } from "@/types/files";

export type ChatComposerState = {
  draft: string;
  setDraft: (draft: string) => void;
  attachedFiles: AttachedFile[];
  attachFile: (file: AttachedFile) => void;
  removeFile: (fileId: string) => void;
  /** A draft typed while the send was in flight survives. */
  clear: (sentText: string) => void;
};

export const useChatComposerState = (): ChatComposerState => {
  const [draft, setDraft] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);

  const attachFile = useCallback((file: AttachedFile) => {
    setAttachedFiles((prev) => [...prev, file]);
  }, []);

  const removeFile = useCallback((fileId: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  const clear = useCallback((sentText: string) => {
    setAttachedFiles([]);
    setDraft((prev) => (prev.trim() === sentText ? "" : prev));
  }, []);

  return { draft, setDraft, attachedFiles, attachFile, removeFile, clear };
};
