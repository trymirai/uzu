import { GlobeIcon } from "lucide-react";
import type { ReactNode } from "react";
import { TextShimmer } from "@/components/ui/text-shimmer";
import type { WebSearchItem } from "@/types/web-search";
import { ToolRow } from "./tool-row";

type WebSearchRowProps = {
  item: WebSearchItem;
  generating: boolean;
  awaitingPermission?: boolean;
};

const resultCount = (count: number): string =>
  count === 0 ? "No results" : count === 1 ? "1 result" : `${count} results`;

const failure = (item: WebSearchItem): string => {
  switch (item.failure?.kind) {
    case "denied":
      return "Search not allowed";
    case "limit":
      return `${item.failure.provider} search limit reached. Choose another provider in Settings.`;
    case "keychain":
      return "Couldn't read the key from the Keychain";
    case "key":
      return `${item.failure.provider} needs a working API key. Add or change it in Settings.`;
    default:
      return "Couldn't search the web";
  }
};

const content = (item: WebSearchItem): ReactNode => {
  if (item.failed) return <span>{failure(item)}</span>;
  if (!item.called) {
    // The query is empty until the model has finished writing the call.
    return (
      <TextShimmer className="truncate">{item.query ? `Searching: ${item.query}` : "Searching the web…"}</TextShimmer>
    );
  }
  return (
    <>
      <span className="shrink-0">Searched</span>
      <span className="min-w-0 truncate text-label-title">{item.query}</span>
      <span className="shrink-0 opacity-80">· {resultCount(item.sources?.length ?? 0)}</span>
    </>
  );
};

export function WebSearchRow({ item, generating, awaitingPermission = false }: WebSearchRowProps) {
  if (!generating && !item.failed) return null;
  if (awaitingPermission && !item.called) return null;

  return (
    <ToolRow>
      <GlobeIcon className="size-3.5 shrink-0" aria-hidden="true" />
      {content(item)}
    </ToolRow>
  );
}
