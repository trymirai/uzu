import { twMerge } from "tailwind-merge";
import { LoaderIcon } from "@/components/loader";
import { platformInfo } from "@/platform/platform-info";

const HEADER_LEFT_RESERVE_PX = platformInfo.features.nativeTitleBar ? 120 : 56;

export const ChatHeader = ({
  title,
  isSidebarOpen,
  isTitleGenerating,
  className,
}: {
  title: string;
  isSidebarOpen: boolean;
  isTitleGenerating?: boolean;
  className?: string;
}) => {
  return (
    <div
      data-tauri-drag-region="deep"
      className={twMerge("flex items-center justify-between pb-4 text-center lg:text-left", className)}
    >
      <div
        className="flex items-center gap-1 w-auto lg:w-auto justify-center lg:justify-start min-w-0"
        style={
          !isSidebarOpen
            ? {
                paddingLeft: `clamp(0px, calc(${HEADER_LEFT_RESERVE_PX}px - var(--sidebar-width, 0px)), ${HEADER_LEFT_RESERVE_PX}px)`,
              }
            : undefined
        }
      >
        <h1
          className={twMerge(
            "text-[13px] font-[350] text-label-muted line-clamp-1",
            !isSidebarOpen && "lg:text-center",
          )}
        >
          {title}
        </h1>
        {isTitleGenerating ? <LoaderIcon width={8} height={8} /> : null}
      </div>
    </div>
  );
};
