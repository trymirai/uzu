import { FileTextIcon, SearchIcon } from "lucide-react";
import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { Spinner } from "@/components/ui/spinner";

type LineIcon = "search" | "page" | "spinner";

export function SearchLine({
  icon,
  failed = false,
  end,
  children,
}: {
  icon: LineIcon;
  failed?: boolean;
  end?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex min-h-[26px] flex-wrap items-start gap-x-2 py-px text-[13px] leading-[19px] text-label-title animate-uzu-in">
      <span className="mt-[5px] grid size-3.5 shrink-0 place-items-center text-label-muted" aria-hidden="true">
        {icon === "spinner" ? (
          <Spinner size={14} />
        ) : icon === "page" ? (
          <FileTextIcon className="size-3.5 animate-uzu-swap" />
        ) : (
          <SearchIcon className="size-3.5 animate-uzu-swap" />
        )}
      </span>
      <span
        className={twMerge(
          "min-w-0 flex-[1_1_220px] py-[2.5px] [overflow-wrap:anywhere] transition-colors duration-150",
          failed && "text-label-muted",
        )}
      >
        {children}
      </span>
      {end && (
        <span className="ml-auto flex items-center gap-0.5 self-center pl-2 text-[12.5px] whitespace-nowrap text-label-muted tabular-nums animate-uzu-swap">
          {end}
        </span>
      )}
    </div>
  );
}

export function Caret() {
  return (
    <span
      aria-hidden="true"
      className="ml-px inline-block h-[15px] w-[1.5px] translate-y-[2.5px] bg-label-title animate-pulse motion-reduce:animate-none"
    />
  );
}

export const typedAddress = (url: string): string => url.replace(/^https?:\/\/(www\.)?/, "");

export function Address({ url }: { url: string }) {
  const shown = typedAddress(url);
  const slash = shown.indexOf("/");
  if (slash === -1) return <>{shown}</>;
  return (
    <>
      {shown.slice(0, slash)}
      <span className="text-label-muted">{shown.slice(slash)}</span>
    </>
  );
}
