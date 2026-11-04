import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { useNavigate, useParams } from "@tanstack/react-router";
import {
  IconDotGrid1x3Horizontal,
  IconPencil,
  IconTrashCan,
} from "@central-icons-react/round-outlined-radius-3-stroke-2";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useChatStore } from "@/stores/use-chat-store";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useShiftHeld } from "@/hooks/use-shift-held";
import { ChatDeleteModal } from "@/features/chat-history/components/chat-delete-modal";
import type { ChatMetadata } from "@/platform/services/storage";

// The row's own fill changes with hover and selection, so its buttons darken whatever is under them
// rather than painting a fixed grey that vanishes on the selected row.
const ROW_BUTTON_CLASSNAME =
  "flex size-6 shrink-0 items-center justify-center rounded-md text-label-muted outline-hidden transition-[opacity,background-color,color] duration-150 ease-out hover:bg-black/[0.07] hover:text-label-title dark:hover:bg-white/[0.08] data-[focus]:shadow-focus";
const MENU_ROW_CLASSNAME =
  "flex w-full min-h-8 items-center gap-2 rounded-lg px-2 text-[13px] font-[400] leading-[1.3] outline-hidden cursor-pointer transition-colors duration-100 ease-out";

const VISIBILITY_THRESHOLDS = Array.from({ length: 101 }, (_, index) => index / 100);

export default function SavedChats() {
  const navigate = useNavigate();
  const { chatId: currentChatId } = useParams({ strict: false }) as {
    chatId?: string;
  };
  const savedChats = useChatStore((s) => s.savedChats);
  const loadSavedChats = useChatStore((s) => s.loadSavedChats);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const updateChatTitle = useChatStore((s) => s.updateChatTitle);
  const setChatVisibility = useSidebarStore((s) => s.setChatVisibility);
  const scrollViewportRef = useRef<HTMLDivElement>(null);
  const currentChatRowRef = useRef<HTMLDivElement>(null);
  const currentChatExists = savedChats.some((chat) => chat.id === currentChatId);

  // Keep the measurement while collapsed so the title and sidebar start
  // their transitions together when reopened.
  useLayoutEffect(() => {
    const root = scrollViewportRef.current;
    const row = currentChatRowRef.current;
    if (!currentChatId) {
      setChatVisibility(null);
      return;
    }
    if (!currentChatExists || !root || !row || typeof IntersectionObserver === "undefined") {
      setChatVisibility({ chatId: currentChatId, ratio: 0 });
      return;
    }
    // The observer's first callback can arrive after paint when entering a chat.
    // Measure now so an already-visible sidebar title never flashes in the header.
    const viewport = root.getBoundingClientRect();
    const bounds = row.getBoundingClientRect();
    const visibleHeight = Math.max(0, Math.min(bounds.bottom, viewport.bottom) - Math.max(bounds.top, viewport.top));
    setChatVisibility({ chatId: currentChatId, ratio: bounds.height > 0 ? visibleHeight / bounds.height : 0 });
    let active = true;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!active) return;
        setChatVisibility({ chatId: currentChatId, ratio: entry?.isIntersecting ? entry.intersectionRatio : 0 });
      },
      { root, threshold: VISIBILITY_THRESHOLDS },
    );
    observer.observe(row);
    return () => {
      active = false;
      observer.disconnect();
      setChatVisibility(null);
    };
  }, [currentChatId, currentChatExists, setChatVisibility]);

  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedChat, setSelectedChat] = useState<ChatMetadata | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  const shiftHeld = useShiftHeld();

  useEffect(() => {
    loadSavedChats();
  }, [loadSavedChats]);

  useEffect(() => {
    if (editingChatId) {
      requestAnimationFrame(() => {
        renameInputRef.current?.focus({ preventScroll: true });
        renameInputRef.current?.select();
      });
    }
  }, [editingChatId]);

  const handleChatClick = (chatId: string) => {
    navigate({ to: "/chat/$chatId", params: { chatId } });
  };

  const startRename = (chat: ChatMetadata) => {
    setEditingChatId(chat.id);
    setDraftTitle(chat.title);
  };

  const cancelRename = () => {
    setEditingChatId(null);
    setDraftTitle("");
  };

  const commitRename = async () => {
    if (!editingChatId) return;
    const chat = savedChats.find((c) => c.id === editingChatId);
    const newTitle = draftTitle.trim();
    cancelRename();
    if (!chat || !newTitle || newTitle === chat.title) {
      return;
    }
    try {
      await updateChatTitle(chat.id, newTitle);
    } catch (error) {
      console.error("Failed to rename chat:", error);
    }
  };

  const performDelete = async (chat: ChatMetadata) => {
    setIsDeleting(true);
    try {
      await deleteChat(chat.id);
      setShowDeleteModal(false);
      setSelectedChat(null);

      if (currentChatId === chat.id) {
        const newChatId = Date.now().toString();
        navigate({
          to: "/chat/$chatId",
          params: { chatId: newChatId },
          search: { isNew: true },
        });
      }
    } catch (error) {
      console.error("Failed to delete chat:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteChat = (chat: ChatMetadata, skipConfirmation: boolean) => {
    if (skipConfirmation) {
      void performDelete(chat);
      return;
    }
    setSelectedChat(chat);
    setShowDeleteModal(true);
  };

  return (
    <div ref={scrollViewportRef} className="flex-1 overflow-y-auto overscroll-y-contain min-h-0 scrollbar-hide">
      <div className="flex flex-col gap-px pb-8">
        {savedChats.length === 0 ? (
          <div className="px-[18px] py-2 text-[14px] font-[400] leading-[1.3] text-label-muted">No chats yet</div>
        ) : (
          savedChats.map((chat: ChatMetadata) => (
            <div
              key={chat.id}
              ref={chat.id === currentChatId ? currentChatRowRef : undefined}
              role="button"
              tabIndex={0}
              aria-label={chat.title}
              aria-current={chat.id === currentChatId ? "page" : undefined}
              onClick={() => {
                if (editingChatId !== chat.id) handleChatClick(chat.id);
              }}
              onKeyDown={(e) => {
                if (e.target !== e.currentTarget) return;
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  if (editingChatId !== chat.id) handleChatClick(chat.id);
                }
              }}
              className="flex px-2"
            >
              <Menu>
                {({ open }) => (
                  <div
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      startRename(chat);
                    }}
                    className={`flex h-9 items-center gap-2 px-2.5 w-full rounded-[10px] group transition-colors duration-150 ease-out ${currentChatId === chat.id ? "bg-sidebar-chat-selected" : editingChatId === chat.id || open ? "bg-sidebar-chat-hover" : "hover:bg-sidebar-chat-hover"}`}
                  >
                    <div className="flex-1 min-w-0">
                      {editingChatId === chat.id ? (
                        <input
                          ref={renameInputRef}
                          value={draftTitle}
                          onChange={(e) => setDraftTitle(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => {
                            e.stopPropagation();
                            if (e.key === "Enter") void commitRename();
                            if (e.key === "Escape") cancelRename();
                          }}
                          onBlur={() => void commitRename()}
                          className="block w-full text-left text-[14px] font-[400] leading-[1.3] text-label-title bg-transparent outline-hidden border-none p-0"
                        />
                      ) : (
                        <p className="w-full text-left text-[14px] font-[400] leading-[1.3] text-label-title truncate">
                          {chat.title}
                        </p>
                      )}
                    </div>
                    {shiftHeld && !open ? (
                      <button
                        type="button"
                        aria-label={`Delete ${chat.title}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          void performDelete(chat);
                        }}
                        onDoubleClick={(e) => e.stopPropagation()}
                        className={`${ROW_BUTTON_CLASSNAME} opacity-0 group-hover:opacity-100 text-danger hover:text-danger`}
                      >
                        <IconTrashCan aria-hidden size={16} />
                      </button>
                    ) : (
                      <>
                        <MenuButton
                          aria-label={`Options for ${chat.title}`}
                          onClick={(e) => e?.stopPropagation()}
                          onDoubleClick={(e) => e.stopPropagation()}
                          className={`${ROW_BUTTON_CLASSNAME} ${open ? "opacity-100 bg-black/[0.07] text-label-title dark:bg-white/[0.08]" : "opacity-0 group-hover:opacity-100 data-[focus]:opacity-100"}`}
                        >
                          <IconDotGrid1x3Horizontal aria-hidden size={16} />
                        </MenuButton>
                        <MenuItems
                          transition
                          anchor="bottom end"
                          className="w-40 rounded-xl bg-surface-popover p-1 shadow-popover [--anchor-gap:4px] origin-top-right transition-[opacity,scale] duration-150 ease-out data-[closed]:scale-95 data-[closed]:opacity-0 data-[leave]:duration-100 motion-reduce:transition-none outline-hidden z-50 translate-x-1"
                        >
                          <MenuItem>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                startRename(chat);
                              }}
                              className={`${MENU_ROW_CLASSNAME} text-text-primary data-[focus]:bg-selection-hover`}
                            >
                              <IconPencil aria-hidden size={16} className="shrink-0 text-text-secondary" />
                              Rename
                            </button>
                          </MenuItem>
                          <MenuItem>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteChat(chat, e.shiftKey || shiftHeld);
                              }}
                              className={`${MENU_ROW_CLASSNAME} text-danger data-[focus]:bg-danger-bg`}
                            >
                              <IconTrashCan aria-hidden size={16} className="shrink-0" />
                              Delete
                            </button>
                          </MenuItem>
                        </MenuItems>
                      </>
                    )}
                  </div>
                )}
              </Menu>
            </div>
          ))
        )}

        {selectedChat && (
          <ChatDeleteModal
            isOpen={showDeleteModal}
            onClose={() => {
              setShowDeleteModal(false);
              setSelectedChat(null);
            }}
            onConfirm={async () => {
              if (selectedChat) await performDelete(selectedChat);
            }}
            chatName={selectedChat.title}
            isDeleting={isDeleting}
          />
        )}
      </div>
    </div>
  );
}
