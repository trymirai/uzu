import { twMerge } from "tailwind-merge";
import { LoaderIcon } from "@/components/loader";
import { platformInfo } from "@/platform/platform-info";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useParams } from "@tanstack/react-router";
import { UNTITLED_CHAT_TITLE } from "@/constants/chat";

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
  const { chatId } = useParams({ strict: false }) as { chatId?: string };
  const chatVisibility = useSidebarStore((s) => s.chatVisibility);
  const sidebarVisibility = isSidebarOpen
    ? chatVisibility && chatVisibility.chatId === chatId
      ? chatVisibility.ratio
      : 1
    : 0;
  // Fade smoothly between 15% and 75% clipped, staying flat at either end.
  const fadeProgress = Math.max(0, Math.min(1, (0.85 - sidebarVisibility) / 0.6));
  const opacity = title === UNTITLED_CHAT_TITLE ? 0 : fadeProgress * fadeProgress * (3 - 2 * fadeProgress);
  return (
    <div
      data-tauri-drag-region="deep"
      className={twMerge("flex items-center justify-between pb-4 text-center lg:text-left", className)}
    >
      <div
        aria-hidden={opacity === 0}
        className="flex items-center gap-1 w-auto lg:w-auto justify-center lg:justify-start min-w-0 transition-[padding-left,opacity] duration-300 ease-in-out motion-reduce:transition-none"
        style={{ paddingLeft: isSidebarOpen ? 0 : HEADER_LEFT_RESERVE_PX, opacity }}
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
