import { GlobeIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Collapse } from "@/components/ui/collapse";
import { TextShimmer } from "@/components/ui/text-shimmer";
import { WEB_SEARCH_PROVIDERS } from "@/constants/web-search-providers";
import { useSettingsStore } from "@/stores/use-settings-store";
import type { WebSearchFailure, WebSearchItem } from "@/types/web-search";
import { useTypedLines } from "../../hooks/use-typed-lines";
import { Caret, SearchLine } from "./search-field";
import { ExpandRow, SourceLinks } from "./source-links";
import { ToolRow } from "./tool-row";

const SETTLE_DELAY_MS = 500;

type WebSearchBatchProps = {
  items: WebSearchItem[];
  generating: boolean;
  awaitingPermission: boolean;
};

const resultCount = (count: number): string =>
  count === 0 ? "No results" : count === 1 ? "1 result" : `${count} results`;

type FailureText = { short: string; full: (provider: string) => string };

const FAILURES: Record<WebSearchFailure["kind"], FailureText> = {
  denied: { short: "Not allowed", full: () => "Search not allowed" },
  limit: {
    short: "Limit reached",
    full: (provider) => `${provider} search limit reached. Choose another provider in Settings.`,
  },
  keychain: { short: "Key unreadable", full: () => "Couldn't read the key from the Keychain" },
  key: {
    short: "No API key",
    full: (provider) => `${provider} needs a working API key. Add or change it in Settings.`,
  },
};

const UNKNOWN_FAILURE: FailureText = { short: "Couldn't search", full: () => "Couldn't search the web" };

const failureOf = ({ failure }: WebSearchItem): { short: string; full: string } => {
  const text = failure ? FAILURES[failure.kind] : UNKNOWN_FAILURE;
  return { short: text.short, full: text.full(failure && "provider" in failure ? failure.provider : "") };
};

const lineEnd = (item: WebSearchItem): string | undefined =>
  item.failed ? failureOf(item).short : item.called ? resultCount(item.sources?.length ?? 0) : undefined;

export function WebSearchBatch({ items, generating, awaitingPermission }: WebSearchBatchProps) {
  const selected = WEB_SEARCH_PROVIDERS[useSettingsStore((state) => state.webSearchProvider)].label;
  const recorded = items.find((item) => item.provider)?.provider;
  const provider = recorded ?? selected;
  const finished = !generating || items.every((item) => item.called || item.failed);
  const [settled, setSettled] = useState(finished);
  const [open, setOpen] = useState(false);
  const typing = useTypedLines(items.map((item) => item.query));

  useEffect(() => {
    if (!finished) {
      setSettled(false);
      return;
    }
    if (settled) return;
    const timer = setTimeout(() => setSettled(true), SETTLE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [finished, settled]);

  if (awaitingPermission) return null;
  if (!generating && !items.some((item) => item.called || item.failed)) return null;

  const found = items.filter((item) => item.called && !item.failed);
  if (settled && found.length === 0) {
    const failed = items.find((item) => item.failed);
    return failed ? (
      <ToolRow>
        <GlobeIcon className="size-3.5 shrink-0" aria-hidden="true" />
        <span>{failureOf(failed).full}</span>
      </ToolRow>
    ) : null;
  }
  const total = found.reduce((sum, item) => sum + (item.sources?.length ?? 0), 0);

  return (
    <div data-tool-call className="mt-1 mb-2 pl-1">
      {settled ? (
        <ExpandRow
          icon={<GlobeIcon className="size-3.5 shrink-0" aria-hidden="true" />}
          open={open}
          onToggle={() => setOpen((value) => !value)}
        >
          {["Searched", recorded, resultCount(total).toLowerCase()].filter(Boolean).join(" · ")}
        </ExpandRow>
      ) : (
        <div className="flex h-6 items-center gap-2 text-[13px] text-label-muted">
          <GlobeIcon className="size-3.5 shrink-0" aria-hidden="true" />
          <TextShimmer>Searching the web</TextShimmer>
          <span className="shrink-0">· {provider}</span>
        </div>
      )}
      <Collapse open={!settled}>
        <div className="mt-0.5">
          {items.slice(0, typing.shown).map((item, index) => {
            const typed = typing.isTyped(index);
            // A result lands only once its query is typed, however fast the search was.
            const done = typed && (item.called || item.failed);
            return (
              <SearchLine
                key={index}
                icon={typed && !done ? "spinner" : "search"}
                failed={done && item.failed}
                end={done ? lineEnd(item) : undefined}
              >
                {typing.textOf(index)}
                {typing.isTyping(index) && <Caret />}
              </SearchLine>
            );
          })}
        </div>
      </Collapse>
      <Collapse open={settled && open}>
        <div className="mt-0.5">
          {items.map((item, index) => (
            <div key={index}>
              <SearchLine icon="search" failed={item.failed} end={lineEnd(item)}>
                {item.query}
              </SearchLine>
              {!item.failed && item.sources && <SourceLinks sources={item.sources} className="mb-1 pl-[22px]" />}
            </div>
          ))}
        </div>
      </Collapse>
    </div>
  );
}
