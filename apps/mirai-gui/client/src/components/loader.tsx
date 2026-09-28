import { useAppStore } from "@/stores/use-app-store";
import { useRive } from "@rive-app/react-canvas";
import React, { type HTMLAttributes } from "react";
import { twMerge } from "tailwind-merge";
import { TextShimmer } from "./ui/text-shimmer";

type LoaderProps = {
  showText?: boolean;
  textClassName?: string;
  text?: string;
  iconWidth?: number;
  iconHeight?: number;
} & HTMLAttributes<HTMLDivElement>;

export const LoaderIcon = ({ width = 10, height = 10 }: { width?: number; height?: number }) => {
  const isDarkMode = useAppStore((s) => s.isDarkMode);

  const baseUrl = import.meta.env.BASE_URL || "/";
  const src = isDarkMode ? `${baseUrl}rive/loader-dark-mode.riv` : `${baseUrl}rive/loader-light-mode.riv`;

  const { RiveComponent } = useRive({
    src: src,
    autoplay: true,
  });

  return (
    <div style={{ width, height }}>
      <RiveComponent key={isDarkMode ? "dark" : "light"} style={{ width: "100%", height: "100%", display: "block" }} />
    </div>
  );
};

export const Loader: React.FC<LoaderProps> = ({
  showText = true,
  textClassName,
  className,
  text = "Loading…",
  iconWidth = 10,
  iconHeight = 10,
}) => {
  return (
    <div className={twMerge("flex items-center justify-center gap-2", className)}>
      <LoaderIcon width={iconWidth} height={iconHeight} />
      {showText && text && (
        <div className="flex items-baseline gap-[2px]">
          <TextShimmer
            duration={1.2}
            className={twMerge(
              "text-[13px] font-[350] [--base-color:theme(colors.label-muted.DEFAULT)] dark:[--base-color:theme(colors.label-muted.dark)]",
              textClassName,
            )}
          >
            {text}
          </TextShimmer>
        </div>
      )}
    </div>
  );
};
