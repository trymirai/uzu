import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { Tooltip } from "@/components/ui/tooltip";
import { Text } from "@/components/ui/typography";

export type FamilyCardBadgeTone = "neutral" | "success";
export type FamilyCardBadgeVariant = "outline" | "filled";

export type FamilyCardBadge = {
  label: ReactNode;
  tooltip?: ReactNode;
  tone?: FamilyCardBadgeTone;
  variant?: FamilyCardBadgeVariant;
};

export type FamilyCardProps = {
  vendorIcon: ReactNode;
  familyName: ReactNode;
  vendorName: ReactNode;
  badges?: FamilyCardBadge[];
  onClick?: () => void;
  compact?: boolean;
};

const TONE_CLASSES: Record<FamilyCardBadgeTone, { outline: string; filled: string; text: string }> = {
  neutral: {
    outline: "border-border-outlined bg-transparent",
    filled: "border-border-outlined bg-surface-secondary",
    text: "text-text-muted",
  },
  success: {
    outline: "border-success-border bg-transparent",
    filled: "border-success-border bg-success-bg",
    text: "text-success",
  },
};

function Chip({ badge }: { badge: FamilyCardBadge }) {
  const { label, tooltip, tone = "neutral", variant = "outline" } = badge;
  const tones = TONE_CLASSES[tone];
  const surface = variant === "filled" ? tones.filled : tones.outline;

  const chip = (
    <div
      className={twMerge(
        "flex items-center justify-center h-7 px-2 py-0.5 rounded border-[0.5px] cursor-default",
        surface,
      )}
    >
      <Text
        as="span"
        opticalSize={14}
        className={twMerge("text-[13px] font-[450] leading-[1.3] whitespace-nowrap", tones.text)}
      >
        {label}
      </Text>
    </div>
  );

  if (tooltip) {
    return (
      <Tooltip content={tooltip} side="top">
        {chip}
      </Tooltip>
    );
  }
  return chip;
}

export function FamilyCard({ vendorIcon, familyName, vendorName, badges, onClick, compact = false }: FamilyCardProps) {
  const fallbackAria = typeof familyName === "string" ? `Open ${familyName} models` : undefined;

  const heading = (
    <div className="flex items-center gap-4 min-w-0 flex-1">
      <span className="shrink-0 flex items-center justify-center w-4 h-4 [&>svg]:w-4 [&>svg]:h-4">{vendorIcon}</span>
      <div className="flex items-baseline gap-1.5 min-w-0">
        <Text as="span" size="sm" color="primary" opticalSize={14} className="truncate leading-[1.5] font-[450]">
          {familyName}
        </Text>
        <Text as="span" size="sm" color="muted" opticalSize={14} className="truncate leading-[1.5] font-[450]">
          from {vendorName}
        </Text>
      </div>
    </div>
  );

  const chips = badges?.map((badge, idx) => <Chip key={idx} badge={badge} />);

  const chevron = <ChevronRight size={14} className="text-text-muted shrink-0" />;

  if (compact) {
    return (
      <button
        type="button"
        onClick={onClick}
        aria-label={fallbackAria}
        className="w-full flex flex-col items-stretch gap-3 px-4 py-3 rounded-lg border-[0.5px] border-border-default bg-surface-elevated transition-colors duration-150 cursor-pointer outline-hidden focus-visible:shadow-focus hover:bg-surface-tertiary text-left"
      >
        <div className="flex items-center gap-3 min-w-0">
          {heading}
          {chevron}
        </div>
        {chips && chips.length > 0 && <div className="flex items-center gap-1.5 flex-wrap">{chips}</div>}
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={fallbackAria}
      className="w-full flex items-center justify-between gap-4 h-12 pl-4 pr-3 rounded-lg border-[0.5px] border-border-default bg-surface-elevated transition-colors duration-150 cursor-pointer outline-hidden focus-visible:shadow-focus hover:bg-surface-tertiary"
    >
      {heading}
      <div className="flex items-center gap-1.5 shrink-0">
        {chips}
        <span className="ml-1">{chevron}</span>
      </div>
    </button>
  );
}
