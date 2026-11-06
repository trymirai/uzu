import { useEffect, useRef, useState } from "react";
import { useIsMobile } from "@/hooks/use-media-query";
import { useNativeFullscreen } from "@/hooks/use-native-fullscreen";
import { platformInfo } from "@/platform/platform-info";
import { useChatStore } from "@/stores/use-chat-store";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useLocation, useNavigate } from "@tanstack/react-router";
import { ChatsIcon, ModelsIcon, NewChatIcon, SidebarIcon } from "./nav-icons";
import { twMerge } from "tailwind-merge";
import { v4 as uuidv4 } from "uuid";
import MenuItem from "./menu-item";
import SavedChats from "@/features/chat-history/components/saved-chats";
import Settings from "./settings";
import { FooterBar } from "../footer-bar";
import { SidebarResizer } from "./sidebar-resizer";

function LeftBar() {
  const location = useLocation();
  const navigate = useNavigate();
  const createNewChat = useChatStore((s) => s.createNewChat);
  const isMobile = useIsMobile();
  const isFullscreen = useNativeFullscreen();

  const isSidebarOpen = useSidebarStore((s) => s.isOpen);
  const toggleSidebar = useSidebarStore((s) => s.toggle);
  const closeOnMobile = useSidebarStore((s) => s.closeOnMobile);
  const sidebarWidth = useSidebarStore((s) => s.width);
  const [resizing, setResizing] = useState(false);

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
      icon: NewChatIcon,
      title: "New chat",
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

  // Narrow windows: the sidebar floats over the content instead of pushing it. Hovering the button
  // peeks it in, leaving lets it go again after a moment; a click keeps it open.
  const [peek, setPeek] = useState(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdPeek = () => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = null;
  };
  const releasePeek = () => {
    holdPeek();
    closeTimer.current = setTimeout(() => setPeek(false), 220);
  };
  useEffect(() => () => holdPeek(), []);
  useEffect(() => setPeek(false), [location.pathname, isMobile]);
  const floating = isMobile;
  const visible = isSidebarOpen || (floating && peek);
  useEffect(() => {
    if (!floating || !visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setPeek(false);
      closeOnMobile();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [floating, visible, closeOnMobile]);

  const toggleSidebarButton = (
    <button
      type="button"
      aria-label={visible ? "Close sidebar" : "Open sidebar"}
      aria-expanded={visible}
      aria-controls="sidebar"
      onClick={() => {
        if (floating && peek && !isSidebarOpen) {
          // Already showing from the hover: the click keeps it.
          setPeek(false);
          toggleSidebar();
          return;
        }
        setPeek(false);
        toggleSidebar();
      }}
      onPointerEnter={() => {
        if (!floating || isSidebarOpen) return;
        holdPeek();
        setPeek(true);
      }}
      onPointerLeave={() => {
        if (floating && peek) releasePeek();
      }}
      className={twMerge(
        "overlay-button flex size-8 items-center justify-center rounded-lg hover:bg-bg-hover transition-[background-color,translate] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none absolute",
        // The same place at every window width. In the Mac app it sits beside the traffic lights;
        // on the web, and in full screen where macOS hides the lights, it lines up with the sidebar
        // icons (its icon starts 18px in, like theirs), or 8px in from the content panel's corner when
        // the sidebar is out of the way.
        platformInfo.features.nativeTitleBar && !isFullscreen
          ? "top-2.5 translate-x-[90px]"
          : !floating && isSidebarOpen
            ? "top-4 left-3 translate-x-0"
            : "top-4 left-3 translate-x-1",
      )}
    >
      {/* The sidebar's real state, not a hover peek. In narrow windows, where hovering opens the
          floating sidebar, the icon always keeps its divider and doesn't change on tap. */}
      <SidebarIcon size={20} open={floating || isSidebarOpen} className="text-label-title" />
    </button>
  );

  return (
    <div
      className={twMerge(
        "relative z-30 h-full min-h-0 shrink-0 motion-reduce:transition-none",
        // Follows the pointer while its edge is dragged; glides when toggled.
        !resizing && "transition-[width] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]",
      )}
      style={{ width: !floating && isSidebarOpen ? sidebarWidth : 0 }}
    >
      {toggleSidebarButton}
      {!floating && <SidebarResizer onDraggingChange={setResizing} />}

      {/* Opened with a click: clicking anywhere else puts it away. No dimming, as in Claude. */}
      {floating && isSidebarOpen && <div className="fixed inset-0 z-40" onClick={closeOnMobile} aria-hidden />}

      <div
        id="sidebar"
        ref={(element) => {
          if (element) element.inert = !visible;
        }}
        aria-hidden={!visible}
        style={floating ? undefined : { width: sidebarWidth }}
        data-tauri-drag-region
        onPointerEnter={() => {
          if (floating && peek) holdPeek();
        }}
        onPointerLeave={() => {
          if (floating && peek) releasePeek();
        }}
        className={twMerge(
          "relative min-h-0 bg-bg-sidebar flex flex-col pt-2 overflow-clip motion-reduce:transition-none",
          floating
            ? "fixed top-2 left-2 bottom-2 w-[280px] z-50 rounded-2xl shadow-[0_0_0_0.5px_var(--color-border-default),0_24px_48px_-12px_rgb(0_0_0/0.28),0_8px_16px_-8px_rgb(0_0_0/0.12)] transition-[translate,opacity]"
            : "h-full transition-transform",
          visible
            ? "translate-x-0 opacity-100 duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]"
            : floating
              ? "-translate-x-[calc(100%+1rem)] opacity-0 duration-200 ease-[cubic-bezier(0.32,0.72,0,1)]"
              : "-translate-x-full duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]",
        )}
      >
        <div className="flex-1 min-h-0 flex flex-col overflow-clip mt-12">
          <div className="flex-shrink-0">
            <nav className="flex flex-col gap-px">
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

            <h2 className="px-[18px] pt-5 pb-1.5 text-[14px] font-[500] leading-[1.3] text-label-muted">Recent</h2>
          </div>

          <SavedChats />

          {/* The footer floats over the end of the chat list: the whole block is one fade, the
              sidebar colour melting to clear upward with a blur that fades the same way, so chats
              scroll down behind it. The settings button sits 12px in from the left and the bottom,
              with as much room above it. */}
          <div className="relative z-10 -mt-14 shrink-0">
            {/* Two layers: the blur fades out through a mask, the colour on top of it unmasked, so
                the lower part stays fully solid behind the buttons. The fade starts 40px above the
                block and turns solid 34px from the bottom, around the button's middle. */}
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 bottom-0 -top-10 backdrop-blur-[3px] [mask-image:linear-gradient(to_top,black_34px,transparent)]"
            />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-x-0 bottom-0 -top-10 bg-[linear-gradient(to_top,var(--color-bg-sidebar)_34px,transparent)]"
            />
            <FooterBar leading={<Settings />} className="relative h-14" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default LeftBar;
