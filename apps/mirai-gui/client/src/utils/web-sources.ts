import type { TranscriptItem } from "@/types/llm-stream";
import type { WebSearchSource } from "@/types/web-search";

const pageKey = (url: string): string => {
  try {
    const parsed = new URL(url);
    return `${parsed.host.toLowerCase()}${parsed.pathname.replace(/\/+$/, "")}${parsed.search}`;
  } catch {
    return url;
  }
};

const sourcesOf = (item: TranscriptItem): WebSearchSource[] =>
  item.type === "webSearch" ? (item.sources ?? []) : item.type === "webPage" && item.source ? [item.source] : [];

export const collectSources = (transcript: TranscriptItem[] | undefined): WebSearchSource[] => {
  const all = (transcript ?? []).flatMap(sourcesOf);
  return all.filter((source, index) => all.findIndex((other) => pageKey(other.url) === pageKey(source.url)) === index);
};
