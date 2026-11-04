import { useFooterRuntimeModel } from "./use-footer-runtime-model";
import { useAppStore } from "@/stores/use-app-store";
import React from "react";
import { twMerge } from "tailwind-merge";
import { EjectIcon } from "../icons/eject-icon";
import { TextShimmer } from "../ui/text-shimmer";
import { UzuCat } from "../uzu-cat";
import { Tooltip } from "../ui/tooltip";

export const FooterBar = React.memo(({ className }: { className?: string }) => {
  const version = useAppStore((s) => s.appVersion);
  const footerModel = useFooterRuntimeModel();

  if (!footerModel.visible && !version) return null;

  // The status line at the foot of the sidebar: the loaded model's state and its one action on the
  // left, the version on the right. The model's full name lives in the tooltip.
  return (
    <div className={twMerge("h-9 shrink-0 flex items-center justify-between gap-2 px-3 select-none", className)}>
      {!footerModel.visible ? <span /> : null}
      {footerModel.visible ? (
        footerModel.status === "loading" || footerModel.status === "ejecting" ? (
          <span key={footerModel.status} className="flex h-6 items-center gap-1.5 px-1.5 animate-uzu-swap">
            <UzuCat
              state={footerModel.status === "ejecting" ? "freeing" : "loading"}
              size={16}
              className="text-text-primary"
            />
            <TextShimmer duration={1.6} className="text-[12px] font-[450] [--base-color:var(--color-label-muted)]">
              {footerModel.status === "ejecting" ? "Ejecting…" : "Loading…"}
            </TextShimmer>
          </span>
        ) : (
          <Tooltip content={`Eject ${footerModel.label}`} side="top">
            <button
              key="resident"
              type="button"
              onClick={footerModel.onEject}
              disabled={!footerModel.canEject}
              aria-label={`Eject ${footerModel.label}`}
              className="flex h-6 items-center gap-1.5 rounded-md px-1.5 text-[12px] font-[450] leading-[1.3] text-text-secondary transition-colors duration-150 ease-out hover:bg-sidebar-chat-hover hover:text-text-primary disabled:opacity-50 disabled:cursor-not-allowed outline-hidden focus-visible:shadow-focus animate-uzu-swap"
            >
              <EjectIcon className="shrink-0" />
              <span>Eject model</span>
            </button>
          </Tooltip>
        )
      ) : null}
      {version ? (
        <div className="shrink-0 px-1.5 text-[11px] font-[450] leading-[1.3] text-text-muted tabular-nums">
          {version}
        </div>
      ) : null}
    </div>
  );
});
