import type { ReactNode } from "react";

export type ModelTableProps = {
  title: ReactNode;
  children: ReactNode;
  compact?: boolean;
};

const HEADER_CELL_CLS = "text-[11px] font-medium uppercase tracking-[0.06em] text-text-muted";

const GRID_TEMPLATE = "grid grid-cols-[minmax(0,1fr)_80px_200px_120px] gap-3 pl-4 pr-3";
const GRID_TEMPLATE_COMPACT = "grid grid-cols-[minmax(0,1fr)_auto] gap-3 pl-4 pr-3";

export function ModelTable({ title, children, compact = false }: ModelTableProps) {
  const firstHeader = typeof title === "string" ? `${title} models` : title;
  const gridTemplate = compact ? GRID_TEMPLATE_COMPACT : GRID_TEMPLATE;

  return (
    <div className="flex flex-col gap-2">
      <div className={`${gridTemplate} pb-1`}>
        <span className={HEADER_CELL_CLS}>{firstHeader}</span>
        {!compact && <span className={HEADER_CELL_CLS}>Size</span>}
        {!compact && <span className={HEADER_CELL_CLS}>Quantization</span>}
        <span aria-hidden />
      </div>

      <div className="rounded-md border-[0.5px] border-border-default overflow-clip divide-y-[0.5px] divide-border-default bg-surface-elevated">
        {children}
      </div>
    </div>
  );
}
