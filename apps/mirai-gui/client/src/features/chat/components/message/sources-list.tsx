import type { WebSearchSource } from "@/types/web-search";
import { siteOf } from "@/utils/site-of";

const SOURCES_SELECTOR = "[data-sources]";
const HEADING = "Sources";

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

export function SourcesList({ sources }: { sources: WebSearchSource[] }) {
  if (sources.length === 0) return null;

  return (
    <section aria-label={HEADING} data-sources className="mt-3.5 border-t border-cell-border pt-2.5">
      <h3 className="mb-1 flex h-5 items-center text-[12px] font-[450] text-label-muted">{HEADING}</h3>
      <ul className="flex flex-col gap-0.5">
        {sources.map((source) => (
          <li key={source.url} className="min-w-0">
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer"
              title={source.url}
              className="group flex min-w-0 items-center gap-2 rounded-md py-[5px] text-[13px] leading-[1.3] outline-hidden focus-visible:shadow-focus"
            >
              <span className="min-w-0 truncate font-[350] text-blue group-hover:underline">{source.title}</span>
              <span className="shrink-0 text-[12px] text-label-muted">{siteOf(source.url)}</span>
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}
