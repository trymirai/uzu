import { useFooterRuntimeModel } from "./use-footer-runtime-model";
import { useAppStore } from "@/stores/use-app-store";
import React, { type ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { EjectIcon } from "../icons/eject-icon";
import { TextShimmer } from "../ui/text-shimmer";
import { UzuCat } from "../uzu-cat";
import { Tooltip } from "../ui/tooltip";
import { ModelName, splitModelName } from "../model-name";

// The loaded model, named like the composer's model pill: the model, then its quantization (muted
// in the tooltip).
const shortModelName = (name: string) => {
  const { base, variant } = splitModelName(name);
  return variant ? `${base} ${variant}` : base;
};

export const FooterBar = React.memo(({ className, leading }: { className?: string; leading?: ReactNode }) => {
  const version = useAppStore((s) => s.appVersion);
  const footerModel = useFooterRuntimeModel();

  if (!leading && !footerModel.visible && !version) return null;

  // The foot of the sidebar: whatever leads on the left (the Settings gear), then on the right the
  // loaded model's state with its one action, and the version. The model's full name lives in the
  // tooltip.
  return (
    <div className={twMerge("h-11 shrink-0 flex items-center justify-between gap-2 px-3 select-none", className)}>
      {leading ?? <span />}
      <div className="flex min-w-0 items-center gap-1">
        {footerModel.visible ? (
          footerModel.status === "loading" || footerModel.status === "ejecting" ? (
            <span key={footerModel.status} className="flex h-6 items-center gap-1.5 px-1.5 animate-uzu-swap">
              <UzuCat
                state={footerModel.status === "ejecting" ? "freeing" : "loading"}
                size={16}
                className="text-text-primary"
              />
              <TextShimmer duration={1.6} className="text-[11px] font-[450] leading-[1.3]">
                {footerModel.status === "ejecting" ? "Ejecting…" : "Loading…"}
              </TextShimmer>
            </span>
          ) : (
            <Tooltip
              content={
                <span className="flex items-baseline gap-1">
                  Eject
                  <ModelName name={footerModel.label} />
                </span>
              }
              side="top"
            >
              <button
                key="resident"
                type="button"
                onClick={footerModel.onEject}
                disabled={!footerModel.canEject}
                aria-label={`Eject ${shortModelName(footerModel.label)}`}
                className="flex h-6 items-center gap-1.5 whitespace-nowrap rounded-md px-1.5 text-[11px] font-[450] leading-[1.3] text-text-muted transition-colors duration-150 ease-out hover:bg-sidebar-chat-hover hover:text-text-primary disabled:opacity-50 disabled:cursor-not-allowed outline-hidden focus-visible:shadow-focus animate-uzu-swap"
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
    </div>
  );
});
