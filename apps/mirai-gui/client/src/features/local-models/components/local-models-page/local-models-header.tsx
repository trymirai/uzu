import { ChevronLeft, MonitorSmartphone } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import type { FamilyEntry } from "../../lib/view";
import { IconAction } from "@/components/ui/icon-action";
import { SearchInput } from "@/components/ui/search-input";
import { Tooltip } from "@/components/ui/tooltip";
import { ModelVendorIcon } from "@/components/model-vendor-icon";

const TEXT_14 = "text-[14px] font-[450] leading-[1.5]";
const TEXT_14_STYLE = { fontVariationSettings: "'opsz' 20" } as const;

type LocalModelsHeaderProps = {
  selectedFamily?: FamilyEntry;
  isMobile: boolean;
  searchQuery: string;
  modelsSearch: string;
  onSearchQueryChange: (value: string) => void;
  onModelsSearchChange: (value: string) => void;
  onBack: () => void;
};

export function LocalModelsHeader({
  selectedFamily,
  isMobile,
  searchQuery,
  modelsSearch,
  onSearchQueryChange,
  onModelsSearchChange,
  onBack,
}: LocalModelsHeaderProps) {
  const search = selectedFamily
    ? { value: modelsSearch, onChange: onModelsSearchChange, placeholder: "Search models" }
    : { value: searchQuery, onChange: onSearchQueryChange, placeholder: "Search families" };

  return (
    <>
      <PageHeader
        title={
          <div data-tauri-drag-region="deep" className="flex h-[52px] shrink-0 items-center gap-2 px-4 min-w-0">
            {selectedFamily ? (
              <>
                <Tooltip content="Back to families" side="bottom">
                  <span className="shrink-0">
                    <IconAction label="Back to families" icon={<ChevronLeft size={16} />} onClick={onBack} />
                  </span>
                </Tooltip>
                <ModelVendorIcon vendor={selectedFamily.vendor} size={16} className="h-4 w-4" />
                <span className={`${TEXT_14} text-text-primary truncate`} style={TEXT_14_STYLE}>
                  {selectedFamily.familyName}
                </span>
                {!isMobile && (
                  <span className={`${TEXT_14} text-text-muted truncate`} style={TEXT_14_STYLE}>
                    from {selectedFamily.vendor}
                  </span>
                )}
              </>
            ) : (
              <>
                <MonitorSmartphone size={16} className="text-text-primary shrink-0" />
                <span className={`${TEXT_14} text-text-primary truncate`} style={TEXT_14_STYLE}>
                  Choose a model to chat
                </span>
              </>
            )}
            <div className="flex-1" />
            {!isMobile && (
              <SearchInput
                value={search.value}
                onChange={search.onChange}
                placeholder={search.placeholder}
                className="w-[260px] shrink-0"
              />
            )}
          </div>
        }
      />
      {isMobile && (
        <div
          className="flex shrink-0 items-center gap-2 px-4 py-2"
          style={{ borderTop: "0.5px solid var(--color-border-default)" }}
        >
          <SearchInput
            value={search.value}
            onChange={search.onChange}
            placeholder={search.placeholder}
            className="min-w-0 flex-1"
          />
        </div>
      )}
    </>
  );
}
