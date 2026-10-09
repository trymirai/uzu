import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { useNavigate, useParams } from "@tanstack/react-router";
import { Edit3, MoreHorizontal, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useChatStore } from "@/stores/use-chat-store";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { CHAT_TITLE_MAX_LENGTH, CHAT_TITLE_MIN_LENGTH } from "@/constants/chat";
import { useShiftHeld } from "@/hooks/use-shift-held";
import { ChatDeleteModal } from "@/features/chat-history/components/chat-delete-modal";
import type { ChatMetadata } from "@/platform/services/storage";

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
  useEffect(() => {
    setChatVisibility(null);
    const root = scrollViewportRef.current;
    const row = currentChatRowRef.current;
    if (!currentChatId || !currentChatExists || !root || !row || typeof IntersectionObserver === "undefined") return;
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
    if (
      !chat ||
      newTitle === chat.title ||
      newTitle.length < CHAT_TITLE_MIN_LENGTH ||
      newTitle.length > CHAT_TITLE_MAX_LENGTH
    ) {
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
      <div className="flex flex-col gap-2">
        {savedChats.length === 0 ? (
          <div className="px-4 py-2 text-[13px] font-[350] leading-[150%] text-label-muted">No chats yet</div>
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
                    className={`flex items-center gap-3 py-[6px] px-2 w-full rounded-md group transition-colors duration-150 ${currentChatId === chat.id ? "bg-sidebar-chat-selected" : editingChatId === chat.id || open ? "bg-sidebar-chat-hover" : "hover:bg-sidebar-chat-hover"}`}
                  >
                    <div className="flex-1 min-w-0">
                      {editingChatId === chat.id ? (
                        <input
                          ref={renameInputRef}
                          value={draftTitle}
                          maxLength={CHAT_TITLE_MAX_LENGTH}
                          onChange={(e) => setDraftTitle(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => {
                            e.stopPropagation();
                            if (e.key === "Enter") void commitRename();
                            if (e.key === "Escape") cancelRename();
                          }}
                          onBlur={() => void commitRename()}
                          className="block w-full text-left text-[13px] font-[350] leading-[150%] text-label-title bg-transparent outline-hidden border-none p-0"
                        />
                      ) : (
                        <p className="w-full text-left text-[13px] font-[350] leading-[150%] text-label-title truncate">
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
                        className="opacity-0 group-hover:opacity-100 hover:bg-bg-hover p-1 rounded transition-opacity"
                      >
                        <Trash2 className="w-3 h-3 text-error" />
                      </button>
                    ) : (
                      <>
                        <MenuButton
                          aria-label={`Options for ${chat.title}`}
                          onClick={(e) => e?.stopPropagation()}
                          onDoubleClick={(e) => e.stopPropagation()}
                          className={`${open ? "opacity-100 bg-bg-hover dark:bg-bg-hover" : "opacity-0 group-hover:opacity-100 data-[focus]:opacity-100"} hover:bg-bg-hover p-1 rounded transition-opacity outline-hidden data-[focus]:shadow-focus`}
                        >
                          <MoreHorizontal className="w-3 h-3 text-label-muted" />
                        </MenuButton>
                        <MenuItems
                          anchor="bottom end"
                          className="w-32 rounded-lg border border-cell-border bg-bg-modal [--anchor-gap:4px] outline-hidden focus:outline-hidden focus:ring-0 z-50 translate-x-2"
                        >
                          <MenuItem>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                startRename(chat);
                              }}
                              className="group/item flex w-full p-1 outline-hidden"
                            >
                              <div className="flex grow rounded-md py-1.5 px-2 gap-2 w-full items-center hover:bg-card-hover group-data-[focus]/item:bg-card-hover transition-colors">
                                <Edit3 className="w-3 h-3 text-label-title" />
                                <span className="text-xs text-label-title">Rename</span>
                              </div>
                            </button>
                          </MenuItem>
                          <MenuItem>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteChat(chat, e.shiftKey || shiftHeld);
                              }}
                              className="group/item flex w-full p-1 outline-hidden"
                            >
                              <div className="flex grow rounded-md py-1.5 px-2 gap-2 w-full items-center hover:bg-card-hover group-data-[focus]/item:bg-card-hover transition-colors">
                                <Trash2 className="w-3 h-3 text-error" />
                                <span className="text-xs text-error">Delete</span>
                              </div>
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
