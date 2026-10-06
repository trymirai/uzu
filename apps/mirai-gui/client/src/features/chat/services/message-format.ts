import type { AttachedFile } from "@/types/files";
import type { ChatRole } from "@/types/chat";
import { Roles } from "@/types/chat";

export const formatMessageWithAttachments = (text: string, files: AttachedFile[]): string =>
  files.length === 0
    ? text
    : files.reduce((acc, file) => acc + `\n\n\`\`\`${file.extension || "txt"}\n${file.content}\n\`\`\``, text);

export type ChatRunMessage = {
  role: ChatRole;
  content: string;
  reasoningContent?: string;
};

export const normalizeMessagesForRun = (messages: ChatRunMessage[]): ChatRunMessage[] => {
  const leading = messages[0];
  const system = leading?.role === Roles.System ? [leading] : [];
  const rest = system.length > 0 ? messages.slice(1) : messages;

  const cleaned = rest.filter((m) =>
    m.role === Roles.User ? true : m.role === Roles.Assistant ? m.content.trim().length > 0 : false,
  );

  const alternated = cleaned.reduce<ChatRunMessage[]>((acc, m) => {
    const prev = acc.at(-1);
    if (!prev) {
      return m.role === Roles.User ? [m] : acc;
    }
    // Two user turns in a row follow a stopped reply; dropping either would lose text.
    return prev.role === m.role
      ? [...acc.slice(0, -1), { ...prev, content: `${prev.content}\n\n${m.content}` }]
      : [...acc, m];
  }, []);

  return system.concat(alternated);
};
