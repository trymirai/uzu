import { FileTextIcon } from "lucide-react";
import { TextShimmer } from "@/components/ui/text-shimmer";
import type { WebPageItem } from "@/types/web-search";
import { siteOf } from "@/utils/site-of";
import { ToolRow } from "./tool-row";

type WebPageRowProps = {
  item: WebPageItem;
  generating: boolean;
};

export function WebPageRow({ item, generating }: WebPageRowProps) {
  if (!generating && !item.failed) return null;

  // The address is empty until the model has finished writing the call.
  const site = item.url ? siteOf(item.url) : "a page";
  return (
    <ToolRow>
      <FileTextIcon className="size-3.5 shrink-0" aria-hidden="true" />
      {item.failed ? (
        <span className="truncate">Couldn&apos;t read {site}</span>
      ) : item.called ? (
        <span className="truncate">Read {site}</span>
      ) : (
        <TextShimmer className="truncate">{`Reading ${site}…`}</TextShimmer>
      )}
    </ToolRow>
  );
}
