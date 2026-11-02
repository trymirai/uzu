import { useFooterRuntimeModel } from "./use-footer-runtime-model";
import { useAppStore } from "@/stores/use-app-store";
import { ModelVendorIcon } from "@/components/model-vendor-icon";
import React from "react";
import { twMerge } from "tailwind-merge";
import { EjectIcon } from "../icons/eject-icon";
import { Loader } from "../loader";

export const FooterBar = React.memo(({ className }: { className?: string }) => {
  const version = useAppStore((s) => s.appVersion);
  const footerModel = useFooterRuntimeModel();

  return (
    <div
      className={twMerge(
        "bg-bg-sidebar min-h-6 shrink-0 border-t border-cell-border flex items-center justify-between gap-3 px-5 py-[5px] select-none",
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-2">
        {footerModel.visible && (
          <>
            {footerModel.status === "loading" || footerModel.status === "ejecting" ? (
              <Loader
                showText
                textClassName="text-[11px]"
                iconWidth={8}
                iconHeight={8}
                text={footerModel.status === "ejecting" ? "Ejecting…" : "Loading model…"}
              />
            ) : (
              <>
                <button
                  onClick={footerModel.onEject}
                  disabled={!footerModel.canEject}
                  className="flex shrink-0 items-center gap-[5px] whitespace-nowrap hover:text-label-title disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <EjectIcon className="text-label-title" />
                  <span className="text-[11px] font-[350] leading-[120%]">Eject model</span>
                </button>
                <span className="flex min-w-0 items-center gap-1.5" title={footerModel.label}>
                  {footerModel.vendor && (
                    <ModelVendorIcon vendor={footerModel.vendor} size={12} className="h-3 w-3 shrink-0" />
                  )}
                  <span className="text-[11px] font-[450] leading-[1.3] truncate text-text-secondary">
                    {footerModel.label}
                  </span>
                </span>
              </>
            )}
          </>
        )}
      </div>
      {version && <div className="shrink-0 text-[11px] font-[450] leading-[1.3] text-text-muted">{version}</div>}
    </div>
  );
});
