import { LinkIcon } from "lucide-react";
import { useState } from "react";
import { Collapse } from "@/components/ui/collapse";
import type { WebSearchSource } from "@/types/web-search";
import { siteOf } from "@/utils/site-of";
import { ExpandRow, SourceLinks } from "./source-links";

const SOURCES_SELECTOR = "[data-sources]";
const HEADING = "Sources";
const NAMED_SITES = 2;

export const formatSourcesForCopy = (
  message: HTMLElement,
  format: "html" | "text",
  sources: WebSearchSource[],
): void => {
  const section = message.querySelector(SOURCES_SELECTOR);
  if (!section) return;
  if (format === "text") {
    const lines = sources.map((source) => `- ${source.title}: ${source.url}`);
    section.replaceWith(document.createTextNode(`\n\n${HEADING}:\n${lines.join("\n")}\n`));
    return;
  }
  const heading = document.createElement("p");
  heading.textContent = `${HEADING}:`;
  const list = document.createElement("ul");
  for (const source of sources) {
    const link = document.createElement("a");
    link.href = source.url;
    link.textContent = source.title;
    list.appendChild(document.createElement("li")).appendChild(link);
  }
  section.replaceWith(heading, list);
};

const summaryOf = (sources: WebSearchSource[]): string => {
  const sites = [...new Set(sources.map((source) => siteOf(source.url)))];
  const named = sites.slice(0, NAMED_SITES);
  const more = sites.length - named.length;
  if (more > 0) return `From ${named.join(", ")} and ${more} more`;
  return `From ${named.join(" and ")}`;
};

export function SourcesList({ sources }: { sources: WebSearchSource[] }) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;

  return (
    <div className="mt-3">
      <ExpandRow
        icon={<LinkIcon className="size-3.5 shrink-0" aria-hidden="true" />}
        open={open}
        onToggle={() => setOpen((value) => !value)}
        ariaLabel={`${HEADING}: ${sources.length}`}
      >
        {summaryOf(sources)}
      </ExpandRow>
      <Collapse open={open}>
        <section aria-label={HEADING} data-sources className="pt-1.5">
          <SourceLinks sources={sources} />
        </section>
      </Collapse>
    </div>
  );
}
