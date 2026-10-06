import { useIsMobile } from "@/hooks/use-media-query";
import { platformInfo } from "@/platform/platform-info";
import { useChatStore } from "@/stores/use-chat-store";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { Transition } from "@headlessui/react";
import { useLocation, useNavigate } from "@tanstack/react-router";
import { Plus } from "lucide-react";
import { useCallback, useEffect } from "react";
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
  const setMobile = useSidebarStore((s) => s.setMobile);
  const closeOnMobile = useSidebarStore((s) => s.closeOnMobile);

  useEffect(() => {
    setMobile(isMobile);
  }, [isMobile, setMobile]);

  const setSidebarWidth = useCallback(
    (open: boolean) => {
      document.documentElement.style.setProperty("--sidebar-width", open && !isMobile ? "200px" : "0px");
    },
    [isMobile],
  );

  useEffect(() => {
    setSidebarWidth(useSidebarStore.getState().isOpen);
  }, [setSidebarWidth]);

  const mainItems = [
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
    {
      icon: ChatsIcon,
      title: "Chats",
      url: "/chats",
      isActive: location.pathname.startsWith("/chats"),
    },
    ...(platformInfo.features.localModelDownloads
      ? [
          {
            icon: ModelsIcon,
            title: "Local Models",
            url: "/local-models",
            isActive: location.pathname.startsWith("/local-models"),
          },
        ]
      : []),
  ];

  const toggleSidebarButton = (
    <button
      onClick={toggleSidebar}
      className={twMerge(
        "overlay-button p-1 rounded-lg hover:bg-bg-hover transition-all duration-300 ease-in-out absolute top-[10px]",
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
    <div className="relative z-30">
      {toggleSidebarButton}

      {isMobile && isSidebarOpen && (
        <div className="fixed top-0 right-0 bottom-0 left-[280px] bg-black/50 z-40 md:hidden" onClick={closeOnMobile} />
      )}

      <Transition
        show={isSidebarOpen}
        enter="transition-all duration-300 ease-out"
        enterFrom="opacity-0 -translate-x-full"
        enterTo="opacity-100 translate-x-0"
        leave="transition-all duration-300 ease-in"
        leaveFrom="opacity-100 translate-x-0"
        leaveTo="opacity-0 -translate-x-full"
        beforeEnter={() => setSidebarWidth(true)}
        beforeLeave={() => setSidebarWidth(true)}
        afterLeave={() => setSidebarWidth(false)}
      >
        <div
          data-tauri-drag-region
          className={twMerge(
            "relative min-h-screen bg-bg-sidebar border-[1px] border-cell-border border-b-0 border-t-0 h-screen flex flex-col py-3 overflow-hidden",
            isMobile ? "fixed top-0 left-0 w-[280px] z-50" : "w-[200px]",
          )}
        >
          <div className="flex-1 flex flex-col overflow-hidden mt-10">
            <div className="flex-shrink-0">
              <nav className="flex flex-col gap-2">
                {mainItems.map((item, index) => (
                  <MenuItem
                    key={index}
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

            <div className="flex-1 overflow-y-auto min-h-0 scrollbar-hide">
              <SavedChats />
            </div>

            <div>
              <Settings />
            </div>
          </div>
        </div>
      </Transition>
    </div>
  );
}

export default LeftBar;
