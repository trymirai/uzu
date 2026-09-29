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
        "bg-bg-sidebar max-h-6 min-h-6 border-t border-cell-border flex items-center justify-between px-5 py-[5px] select-none",
        className,
      )}
    >
      <div className="flex items-center gap-2 overflow-hidden">
        {footerModel.visible && (
          <>
            {footerModel.status === "loading" || footerModel.status === "ejecting" ? (
              <Loader
                showText
                textClassName="text-[11px]"
                iconWidth={8}
                iconHeight={8}
                text={footerModel.status === "ejecting" ? "Ejecting…" : "Initializing model…"}
              />
            ) : (
              <>
                <button
                  onClick={footerModel.onEject}
                  disabled={!footerModel.canEject}
                  className="flex items-center gap-[5px] hover:text-label-title disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <EjectIcon className="text-label-title" />
                  <span className="text-[11px] font-[350] leading-[120%]">Eject model</span>
                </button>
                <span className="flex items-center gap-1.5 truncate max-w-[220px]">
                  {footerModel.vendor && <ModelVendorIcon vendor={footerModel.vendor} size={12} className="h-3 w-3" />}
                  <span className="text-[11px] font-[450] leading-[1.3] truncate text-text-secondary">
                    {footerModel.label}
                  </span>
                </span>
              </>
            )}
          </>
        )}
      </div>
      {version && <div className="ml-auto text-[11px] font-[450] leading-[1.3] text-text-muted">{version}</div>}
    </div>
  );
});
