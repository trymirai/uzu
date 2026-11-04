import { ChevronLeft } from "lucide-react";
import { PageTitleBar } from "@/components/page-title-bar";
import type { FamilyEntry } from "../../lib/view";
import { IconAction } from "@/components/ui/icon-action";
import { Tooltip } from "@/components/ui/tooltip";
import { ModelVendorIcon } from "@/components/model-vendor-icon";
import { FamilyBadge } from "../family-card";
import { formatParamRange } from "../../lib/view";

type LocalModelsHeaderProps = {
  selectedFamily?: FamilyEntry;
  isMobile: boolean;
  onBack: () => void;
  /** Slides the title in with the page when moving between the families and a family. */
  animationClassName?: string;
};

/** The page title, set like the chats page: in the column, at the top. A family adds a way back. */
export function LocalModelsHeader({ selectedFamily, isMobile, onBack, animationClassName }: LocalModelsHeaderProps) {
  const range = selectedFamily
    ? formatParamRange(selectedFamily.paramRange?.min, selectedFamily.paramRange?.max)
    : undefined;
  return (
    <PageTitleBar>
      <div
        key={selectedFamily?.familyIdentifier ?? "families"}
        className={`flex min-w-0 flex-1 items-center gap-2 ${animationClassName ?? ""}`}
      >
        {selectedFamily ? (
          <>
            <Tooltip content="Back to local models" side="bottom">
              <span className="shrink-0 -ml-1.5">
                <IconAction label="Back to local models" icon={<ChevronLeft size={18} />} onClick={onBack} />
              </span>
            </Tooltip>
            <ModelVendorIcon vendor={selectedFamily.vendor} size={20} className="size-5" />
            <h1 className="text-[20px] font-[600] leading-[1.2] tracking-[-0.01em] text-label-title truncate">
              {selectedFamily.familyName}
            </h1>
            {/* Who made it and what's inside, as the same pills the family list uses. */}
            <div className="ml-auto flex shrink-0 items-center gap-1.5">
              {[
                { label: selectedFamily.vendor },
                ...(isMobile
                  ? []
                  : [
                      { label: `${selectedFamily.totalCount} ${selectedFamily.totalCount === 1 ? "model" : "models"}` },
                      ...(range ? [{ label: range }] : []),
                    ]),
              ].map((badge) => (
                <FamilyBadge key={badge.label} badge={badge} />
              ))}
            </div>
          </>
        ) : (
          <h1 className="text-[20px] font-[600] leading-[1.2] tracking-[-0.01em] text-label-title">Local models</h1>
        )}
      </div>
    </PageTitleBar>
  );
}
