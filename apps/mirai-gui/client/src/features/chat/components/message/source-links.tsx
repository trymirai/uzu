import { ChevronRightIcon } from "lucide-react";
import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import type { WebSearchSource } from "@/types/web-search";
import { siteOf } from "@/utils/site-of";
import { SiteIcon } from "./site-icon";

export function ExpandRow({
  icon,
  open,
  onToggle,
  ariaLabel,
  children,
}: {
  icon: ReactNode;
  open: boolean;
  onToggle: () => void;
  ariaLabel?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-label={ariaLabel}
      onClick={onToggle}
      className="flex h-6 max-w-full items-center gap-2 rounded-md text-[13px] text-label-muted outline-hidden transition-colors duration-150 hover:text-label-title focus-visible:shadow-focus"
    >
      {icon}
      <span className="min-w-0 truncate">{children}</span>
      <ChevronRightIcon
        aria-hidden="true"
        className={twMerge(
          "size-3.5 shrink-0 transition-transform duration-250 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none",
          open && "rotate-90",
        )}
      />
    </button>
  );
}

export function SourceLinks({ sources, className }: { sources: WebSearchSource[]; className?: string }) {
  return (
    <ul className={twMerge("flex flex-col gap-0.5", className)}>
      {sources.map((source) => (
        <li key={source.url} className="min-w-0">
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer"
            title={source.url}
            className="group flex min-w-0 items-center gap-2 rounded-md py-[5px] text-[13px] leading-[1.3] outline-hidden focus-visible:shadow-focus"
          >
            <SiteIcon url={source.url} />
            <span className="min-w-0 truncate font-[350] text-blue group-hover:underline">{source.title}</span>
            <span className="shrink-0 text-[12px] text-label-muted">{siteOf(source.url)}</span>
          </a>
        </li>
      ))}
    </ul>
  );
}
