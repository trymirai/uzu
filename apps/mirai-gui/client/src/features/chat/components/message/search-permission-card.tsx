import { ChevronRightIcon, GlobeIcon } from "lucide-react";
import { useState } from "react";
import { twMerge } from "tailwind-merge";
import { Button } from "@/components/ui/button";
import { SURFACE_BUTTON_SMALL_CLASSNAME, SURFACE_SOFT_BUTTON_CLASSNAME } from "@/components/ui/modal/constants";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useWebSearchStore } from "@/stores/use-web-search-store";
import type { SearchPermissionRequest } from "@/types/web-search";

// A query or an address is never cut short: the user allows exactly what they can read.
const QUERY_CLASSNAME = "mt-0.5 text-[13px] leading-[1.45] text-label-muted [overflow-wrap:anywhere]";
const EASE_CLASSNAME = "duration-250 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none";

const cardTitle = ({ provider, queries, pages }: SearchPermissionRequest): string => {
  if (pages.length === 0) {
    return queries.length === 1
      ? `Search the web with ${provider}?`
      : `Run ${queries.length} searches with ${provider}?`;
  }
  if (queries.length > 0) return `Search and read pages with ${provider}?`;
  return pages.length === 1 ? `Read this page with ${provider}?` : `Read ${pages.length} pages with ${provider}?`;
};

export function SearchPermissionCard({
  request,
}: {
  request: SearchPermissionRequest & { answering: boolean; chatId: string };
}) {
  const answer = useChatSessionStore((s) => s.answerSearchPermission);
  const alwaysAllow = async () => {
    if (await answer("alwaysAllow")) useWebSearchStore.getState().setChatMode(request.chatId, "auto");
  };
  const [restOpen, setRestOpen] = useState(false);
  const [first, ...rest] = [...request.queries, ...request.pages];
  const title = cardTitle(request);

  return (
    <div
      role="group"
      aria-label={title}
      data-tool-call
      className="mt-1 mb-2 rounded-xl bg-surface-soft py-3 pr-3 pl-3.5 animate-uzu-in"
    >
      <div className="flex items-start gap-2 text-[14px] font-[500] leading-[1.35] text-label-title">
        <GlobeIcon className="mt-[2.5px] size-3.5 shrink-0 text-label-muted" aria-hidden="true" />
        <span>{title}</span>
      </div>
      <div className="ml-[22px]">
        <p className={QUERY_CLASSNAME}>{first}</p>
        {rest.length > 0 && (
          <>
            <Button
              kind="ghost"
              size="xxs"
              aria-expanded={restOpen}
              onClick={() => setRestOpen((open) => !open)}
              icon={
                <ChevronRightIcon
                  className={twMerge("size-3.5 transition-transform", EASE_CLASSNAME, restOpen && "rotate-90")}
                />
              }
              iconPosition="right"
              className="mt-0.5 -ml-2 text-[13px] font-normal text-label-muted"
            >
              {rest.length} more
            </Button>
            <div
              aria-hidden={!restOpen}
              className={twMerge(
                "grid transition-[grid-template-rows,opacity]",
                EASE_CLASSNAME,
                restOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
              )}
            >
              <div className="min-h-0 overflow-hidden">
                {rest.map((query, index) => (
                  <p key={index} className={QUERY_CLASSNAME}>
                    {query}
                  </p>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
      <div className="mt-3 flex flex-wrap justify-end gap-2">
        <Button
          kind="secondary"
          size="sm"
          disabled={request.answering}
          onClick={() => void answer("deny")}
          className={twMerge(SURFACE_BUTTON_SMALL_CLASSNAME, SURFACE_SOFT_BUTTON_CLASSNAME)}
        >
          Deny
        </Button>
        {request.pages.length === 0 && (
          <Button
            kind="secondary"
            size="sm"
            disabled={request.answering}
            onClick={() => void alwaysAllow()}
            className={twMerge(SURFACE_BUTTON_SMALL_CLASSNAME, SURFACE_SOFT_BUTTON_CLASSNAME)}
          >
            Always allow
          </Button>
        )}
        <Button
          kind="primary"
          size="sm"
          disabled={request.answering}
          onClick={() => void answer("allow")}
          className={SURFACE_BUTTON_SMALL_CLASSNAME}
        >
          Allow
        </Button>
      </div>
    </div>
  );
}
