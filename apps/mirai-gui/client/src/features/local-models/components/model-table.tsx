import { twMerge } from "tailwind-merge";
import type { ReactNode } from "react";

export type ModelTableProps = {
  title: ReactNode;
  children: ReactNode;
  compact?: boolean;
};

const HEADER_CELL_CLS = "text-[13px] font-[500] leading-[1.3] text-text-muted whitespace-nowrap truncate";

const GRID_TEMPLATE = "grid grid-cols-[minmax(0,1fr)_128px_64px_96px] gap-3 pl-4 pr-3";
const GRID_TEMPLATE_COMPACT = "grid grid-cols-[minmax(0,1fr)_auto] gap-3 pl-4 pr-3";

export function ModelTable({ title, children, compact = false }: ModelTableProps) {
  const firstHeader = typeof title === "string" ? `${title} models` : title;
  const gridTemplate = compact ? GRID_TEMPLATE_COMPACT : GRID_TEMPLATE;

  return (
    <div className="flex flex-col gap-2">
      {/* The heading starts at the cards' edge, like the settings section headings; the other
          columns are sized from the right, so they still sit over the rows' values. */}
      <div className={twMerge(gridTemplate, "pb-1 pl-1")}>
        <span className={HEADER_CELL_CLS}>{firstHeader}</span>
        {!compact && <span className={HEADER_CELL_CLS}>Quantization</span>}
        {!compact && <span className={HEADER_CELL_CLS}>Size</span>}
        <span aria-hidden />
      </div>

      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}
