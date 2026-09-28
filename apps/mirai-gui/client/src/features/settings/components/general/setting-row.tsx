import type { ReactNode } from "react";

export function SettingDivider() {
  return <div className="h-[1px] bg-cell-border dark:bg-cell-border-dark" />;
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
    <div className="px-5 lg:max-w-[800px] mx-auto w-full">
      <div className="flex items-center justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h4 className="text-[15px] font-[350] leading-[150%] text-label-title dark:text-label-title-dark overflow-hidden">
            {title}
          </h4>
          <p className="text-[13px] text-label-muted dark:text-label-muted-dark">{description}</p>
        </div>
        {control}
      </div>
      {children}
    </div>
  );
}
