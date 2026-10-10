import { useIsMobile } from "@/hooks/use-media-query";
import { platformInfo } from "@/platform/platform-info";
import { useChatStore } from "@/stores/use-chat-store";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useLocation, useNavigate } from "@tanstack/react-router";
import { Plus } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { v4 as uuidv4 } from "uuid";
import { ChatsIcon } from "../icons/chats-icon";
import { ModelsIcon } from "../icons/models-icon";
import SidebarToggleIcon from "../icons/sidebar-toggle-icon";
import MenuItem from "./menu-item";
import SavedChats from "@/features/chat-history/components/saved-chats";
import Settings from "./settings";

function LeftBar() {
  const location = useLocation();
  const navigate = useNavigate();
  const createNewChat = useChatStore((s) => s.createNewChat);
  const isMobile = useIsMobile();

  const isSidebarOpen = useSidebarStore((s) => s.isOpen);
  const toggleSidebar = useSidebarStore((s) => s.toggle);
  const closeOnMobile = useSidebarStore((s) => s.closeOnMobile);

  const mainItems = [
    ...(platformInfo.features.localModelDownloads
      ? [
          {
            icon: ModelsIcon,
            title: "Models",
            url: "/local-models",
            isActive: location.pathname.startsWith("/local-models"),
          },
        ]
      : []),
    {
      icon: ChatsIcon,
      title: "Chats",
      url: "/chats",
      isActive: location.pathname.startsWith("/chats"),
    },
    {
      icon: Plus,
      title: "New Chat",
      onClick: () => {
        const newChatId = uuidv4();
        createNewChat(newChatId);
        navigate({
          to: "/chat/$chatId",
          params: { chatId: newChatId },
          search: { isNew: true },
        });
        closeOnMobile();
      },
    },
  ];

  const toggleSidebarButton = (
    <button
      type="button"
      aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
      aria-expanded={isSidebarOpen}
      aria-controls="sidebar"
      onClick={toggleSidebar}
      className={twMerge(
        "overlay-button p-1 rounded-lg hover:bg-bg-hover transition-colors duration-300 ease-in-out motion-reduce:transition-none absolute top-[10px]",
        isMobile && platformInfo.features.nativeTitleBar
          ? "left-20 translate-x-0"
          : isMobile
            ? "left-4 translate-x-0"
            : platformInfo.features.nativeTitleBar
              ? "translate-x-[92px]"
              : "left-3 translate-x-0",
      )}
    >
      <SidebarToggleIcon className="w-6 h-6 text-label-muted" />
    </button>
  );

  return (
    <div
      className={twMerge(
        "relative z-30 h-full min-h-0 shrink-0 transition-[width] duration-300 ease-in-out motion-reduce:transition-none",
        !isMobile && isSidebarOpen ? "w-62" : "w-0",
      )}
    >
      {toggleSidebarButton}

      {isMobile && isSidebarOpen && (
        <div className="fixed top-0 right-0 bottom-0 left-[280px] bg-black/50 z-40 md:hidden" onClick={closeOnMobile} />
      )}

      <div
        id="sidebar"
        ref={(element) => {
          if (element) element.inert = !isSidebarOpen;
        }}
        aria-hidden={!isSidebarOpen}
        data-tauri-drag-region
        className={twMerge(
          "relative min-h-0 bg-bg-sidebar border-[1px] border-cell-border border-b-0 border-t-0 flex flex-col py-3 overflow-clip transition-transform duration-300 ease-in-out motion-reduce:transition-none",
          isMobile ? "fixed top-0 left-0 h-dvh w-[280px] z-50" : "h-full w-62",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex-1 min-h-0 flex flex-col overflow-clip mt-10">
          <div className="flex-shrink-0">
            <nav className="flex flex-col gap-2">
              {mainItems.map((item) => (
                <MenuItem
                  key={item.title}
                  icon={item.icon}
                  title={item.title}
                  url={item.url}
                  isActive={item.isActive}
                  onClick={item.onClick}
                />
              ))}
            </nav>

            <div className="mt-3 mb-2 border-t border-cell-border" />
          </div>

          <SavedChats />

          <div className="shrink-0">
            <Settings />
          </div>
        </div>
      </div>
    </div>
  );
}

export default LeftBar;
