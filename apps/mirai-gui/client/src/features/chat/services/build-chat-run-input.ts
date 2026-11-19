import type { Message } from "@/types/message";
import type { TranscriptItem } from "@/types/llm-stream";
import { Roles, type NonSystemRole } from "../types";
import type { AttachedFile } from "@/types/files";
import { attachmentStorage } from "./attachment-storage";
import { formatMessageWithAttachments, normalizeMessagesForRun, type ChatRunMessage } from "./message-format";

type HistoryMessage = {
  role: NonSystemRole;
  content: string;
  reasoningContent?: string;
  openableUrls?: string[];
};

const addressesOf = (item: TranscriptItem): string[] => {
  if (item.type === "webSearch") return (item.sources ?? []).map((source) => source.url);
  if (item.type === "webPage" && item.called && !item.failed) {
    return [item.url, item.source?.url].filter((url): url is string => Boolean(url));
  }
  return [];
};

const pageAddresses = (transcript: TranscriptItem[] | undefined): string[] => {
  const all = (transcript ?? []).flatMap(addressesOf);
  return all.filter((url, index) => all.indexOf(url) === index);
};

const projectHistoryMessage = (m: Message): HistoryMessage => {
  const hasVersions = Array.isArray(m.versions) && (m.versions?.length ?? 0) > 0;
  const activeVersion = hasVersions ? m.versions?.[m.currentVersionIndex ?? (m.versions?.length ?? 1) - 1] : undefined;
  const output = hasVersions ? activeVersion?.output : m.output;
  const parsed = output?.text?.parsed;
  const text = hasVersions ? (activeVersion?.text ?? m.text) : m.text;
  const openableUrls = m.sender === Roles.Assistant ? pageAddresses(output?.transcript) : [];
  // History stores attachment ids separately, so reconstruct the content sent to the model.
  const content =
    m.sender === Roles.User && m.attachmentIds?.length
      ? formatMessageWithAttachments(text, attachmentStorage.getFiles(m.attachmentIds))
      : text;
  return {
    role: m.sender,
    content,
    ...(m.sender === Roles.Assistant && parsed?.chainOfThought ? { reasoningContent: parsed.chainOfThought } : {}),
    ...(openableUrls.length ? { openableUrls } : {}),
  };
};

type BuildChatRunInputParams = {
  history: Message[];
  prompt: string;
  attachments: AttachedFile[];
  globalInstructions?: string | null;
};

type BuildChatRunInputResult = {
  messages: ChatRunMessage[];
};

export const buildChatRunInput = (params: BuildChatRunInputParams): BuildChatRunInputResult => {
  const { history, prompt, attachments, globalInstructions } = params;

  const formattedPrompt = formatMessageWithAttachments(prompt, attachments);

  const historyMessages = history
    .map(projectHistoryMessage)
    .filter((m) => !(m.role === Roles.Assistant && m.content.trim().length === 0));

  const base: ChatRunMessage[] = globalInstructions?.trim()
    ? [{ role: Roles.System, content: globalInstructions }, ...historyMessages]
    : historyMessages;

  const preNormalized: ChatRunMessage[] = [...base, { role: Roles.User, content: formattedPrompt }];
  const messages = normalizeMessagesForRun(preNormalized);

  return { messages };
};
