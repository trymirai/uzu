import { ChevronDownIcon, GlobeIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PopoverMenu } from "@/components/ui/popover/popover-menu";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useWebSearchStore } from "@/stores/use-web-search-store";
import type { SearchPermissionRequest } from "@/types/web-search";
import { useTypedLines } from "../../hooks/use-typed-lines";
import { Address, Caret, SearchLine, typedAddress } from "./search-field";

const ACTION_CLASSNAME = "h-6 rounded-full px-2.5 text-[12px] font-[500]";

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
  const items = [
    ...request.queries.map((text) => ({ text, page: false })),
    ...request.pages.map((text) => ({ text, page: true })),
  ];
  const typing = useTypedLines(items.map((item) => (item.page ? typedAddress(item.text) : item.text)));
  const title = cardTitle(request);
  // A written address is always asked about: it can't be allowed for the rest of the chat.
  const canAlwaysAllow = request.pages.length === 0;

  const actions = (
    <span className="flex items-center gap-1">
      <Button
        kind="ghost"
        size="xxs"
        disabled={request.answering}
        onClick={() => void answer("deny")}
        className={ACTION_CLASSNAME}
      >
        Deny
      </Button>
      <span className="flex">
        <Button
          kind="primary"
          size="xxs"
          disabled={request.answering}
          onClick={() => void answer("allow")}
          className={`${ACTION_CLASSNAME} ${canAlwaysAllow ? "rounded-r-none pr-2" : ""}`}
        >
          Allow
        </Button>
        {canAlwaysAllow && (
          <PopoverMenu
            side="bottom"
            align="end"
            sideOffsetPx={6}
            className="min-w-[230px]"
            trigger={
              <Button
                kind="primary"
                size="xxs"
                iconOnly
                ariaLabel="More ways to allow"
                disabled={request.answering}
                icon={<ChevronDownIcon className="size-3.5" aria-hidden="true" />}
                className="h-6 w-auto rounded-l-none rounded-r-full border-l border-black/15 px-1.5"
              />
            }
            items={[
              {
                id: "always",
                label: (
                  <span>
                    <span className="block text-[13px] font-[500] leading-[1.3]">Always allow</span>
                    <span className="mt-0.5 block text-[12px] leading-[1.35] text-label-muted">
                      Search without asking in this chat
                    </span>
                  </span>
                ),
                onClick: () => void alwaysAllow(),
              },
            ]}
          />
        )}
      </span>
    </span>
  );

  return (
    <div role="group" aria-label={title} data-tool-call className="mt-1 mb-2 pl-1 animate-uzu-in">
      <div className="flex h-6 items-center gap-2 text-[13px] text-label-muted">
        <GlobeIcon className="size-3.5 shrink-0" aria-hidden="true" />
        <span>{title}</span>
      </div>
      <div className="mt-0.5">
        {items.slice(0, typing.shown).map((item, index) => (
          <SearchLine
            key={index}
            icon={item.page ? "page" : "search"}
            end={typing.allTyped && index === items.length - 1 ? actions : undefined}
          >
            {item.page && typing.isTyped(index) ? <Address url={item.text} /> : typing.textOf(index)}
            {typing.isTyping(index) && <Caret />}
          </SearchLine>
        ))}
      </div>
    </div>
  );
}
