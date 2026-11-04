import type { ReactNode } from "react";

/** A titled group of settings: a small heading over one soft card, rows split by hairlines. The
 *  card does not clip, so menus opened from a row (the theme picker) can spill past it. */
export function SettingGroup({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-2">
      {title ? <h3 className="px-1 text-[13px] font-[500] leading-[1.3] text-label-muted">{title}</h3> : null}
      <div className="rounded-2xl bg-surface-soft divide-y-[0.5px] divide-border-default [&>*:first-child]:rounded-t-2xl [&>*:last-child]:rounded-b-2xl">
        {children}
      </div>
    </section>
  );
}

export function SettingRow({
  title,
  description,
  control,
  children,
}: {
  title: string;
  description: ReactNode;
  control?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="px-4 py-3.5">
      <div className="flex items-center justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h4 className="text-[14px] font-[500] leading-[1.35] text-label-title">{title}</h4>
          <p className="text-[13px] leading-[1.45] text-label-muted text-pretty">{description}</p>
        </div>
        {control}
      </div>
      {children}
    </div>
  );
}
