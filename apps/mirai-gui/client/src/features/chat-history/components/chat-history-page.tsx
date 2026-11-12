import { PencilIcon } from "@/components/icons/app-icons";
import ChatCard from "./chat-card";
import { ChatDeleteModal } from "./chat-delete-modal";
import { ChatRenameModal } from "./chat-rename-modal";
import { GroupButton } from "@/components/ui/group-button";
import { useNavigate } from "@tanstack/react-router";
import { Plus, Trash2, X } from "lucide-react";
import { v4 as uuidv4 } from "uuid";
import { UzuCat } from "@/components/uzu-cat";
import { PageScroller, PageTitleBar } from "@/components/page-title-bar";
import { type CSSProperties, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { SearchInput } from "@/components/ui/search-input";
import { useToast } from "@/components/ui/toast/use-toast";
import { useChatStore } from "@/stores/use-chat-store";
import { useAppStore } from "@/stores/use-app-store";

export function ChatHistoryPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const chatWidth = useAppStore((s) => s.chatWidth);
  const [searchQuery, setSearchQuery] = useState("");
  const savedChats = useChatStore((s) => s.savedChats);
  const loadSavedChats = useChatStore((s) => s.loadSavedChats);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const createNewChat = useChatStore((s) => s.createNewChat);

  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRenameOpen, setIsRenameOpen] = useState(false);

  useEffect(() => {
    loadSavedChats();
  }, [loadSavedChats]);

  const filteredChats = useMemo(() => {
    const searchLower = searchQuery.toLowerCase();
    return savedChats.filter((chat) => (searchQuery === "" ? true : chat.title.toLowerCase().includes(searchLower)));
  }, [savedChats, searchQuery]);

  const allFilteredIds = filteredChats.map((c) => c.id);
  const isAllSelected = allFilteredIds.length > 0 && allFilteredIds.every((id) => selectedIds.includes(id));

  const toggleSelectAll = () => setSelectedIds(isAllSelected ? [] : allFilteredIds);

  const toggleSelectOne = (chatId: string) =>
    setSelectedIds((prev) => (prev.includes(chatId) ? prev.filter((id) => id !== chatId) : [...prev, chatId]));

  const exitSelectionMode = () => {
    setIsSelectionMode(false);
    setSelectedIds([]);
  };

  const handleChatClick = (chatId: string) => {
    return isSelectionMode
      ? toggleSelectOne(chatId)
      : navigate({
          to: "/chat/$chatId",
          params: { chatId },
          search: { isNew: false },
        });
  };

  const startNewChat = () => {
    const chatId = uuidv4();
    createNewChat(chatId);
    navigate({ to: "/chat/$chatId", params: { chatId }, search: { isNew: true } });
  };

  const confirmBulkDelete = async () => {
    setIsDeleting(true);
    try {
      await Promise.all(selectedIds.map((id) => deleteChat(id)));
    } catch (e) {
      console.error("[chats] bulk delete failed", e);
      toast.error("Some chats could not be deleted");
    } finally {
      setIsDeleting(false);
      setIsConfirmOpen(false);
      exitSelectionMode();
    }
  };

  const singleSelectedChat = useMemo(
    () => (selectedIds.length === 1 ? savedChats.find((c) => c.id === selectedIds[0]) || null : null),
    [selectedIds, savedChats],
  );

  const selectedCount = selectedIds.length;
  const showActive = selectedCount > 0;
  const labelText = isAllSelected ? "All selected" : selectedCount === 0 ? "Select all" : `${selectedCount} selected`;

  return (
    <div className="flex h-full min-h-0 flex-col" style={{ "--chat-width": `${chatWidth}px` } as CSSProperties}>
      <PageScroller
        className="bg-background"
        header={
          <PageTitleBar className="justify-between">
            <h1 className="text-[20px] font-[600] leading-[1.2] tracking-[-0.01em] text-label-title">Chat history</h1>
            {/* Search and selection only once there is something to search or select. */}
            {savedChats.length > 0 ? (
              <div className="flex items-center gap-2">
                <SearchInput
                  value={searchQuery}
                  onChange={setSearchQuery}
                  placeholder="Search"
                  className="w-full max-w-[210px] lg:min-w-[240px] lg:max-w-[240px]"
                />
                {isSelectionMode ? (
                  <GroupButton>
                    <GroupButton.Segment ariaLabel="Select all" onClick={() => toggleSelectAll()}>
                      <Checkbox
                        checked={isAllSelected}
                        onChange={() => {}}
                        size="sm"
                        disabled
                        className="pointer-events-none opacity-100"
                      />
                      <span className={showActive ? "text-text-primary" : undefined}>{labelText}</span>
                    </GroupButton.Segment>

                    {!isAllSelected && singleSelectedChat && (
                      <GroupButton.Segment icon ariaLabel="Rename" onClick={() => setIsRenameOpen(true)}>
                        <PencilIcon size={16} />
                      </GroupButton.Segment>
                    )}

                    <GroupButton.Segment
                      icon
                      ariaLabel="Delete"
                      onClick={() => setIsConfirmOpen(true)}
                      disabled={selectedIds.length === 0}
                    >
                      <Trash2 className="size-4" aria-hidden />
                    </GroupButton.Segment>

                    <GroupButton.Segment icon ariaLabel="Exit selection" onClick={exitSelectionMode}>
                      <X className="size-4" />
                    </GroupButton.Segment>
                  </GroupButton>
                ) : (
                  <Button
                    kind="secondary"
                    size="sm"
                    className="h-8 px-3.5 rounded-full bg-control-fill shadow-none text-[13px] font-[450] text-text-muted hover:bg-control-fill-hover hover:text-text-primary"
                    onClick={() => setIsSelectionMode(true)}
                  >
                    Select
                  </Button>
                )}
              </div>
            ) : null}
          </PageTitleBar>
        }
      >
        <div className="px-5 overflow-visible w-full min-h-full flex flex-col max-w-[calc(var(--chat-width)+2.5rem)] mx-auto">
          {filteredChats.length === 0 && !searchQuery ? (
            <div className="flex flex-1 flex-col items-center justify-center pb-16 text-center animate-uzu-in">
              <div className="mb-1">
                <UzuCat state="resting" size={112} className="text-text-primary" label="uzu, asleep" />
              </div>
              <div className="flex flex-col gap-1">
                <p className="text-[15px] font-[500] text-label-title">No chats yet</p>
                <p className="text-[13px] text-label-muted">Conversations stay on this device</p>
              </div>
              {/* As far below the words as the cat sits above them. */}
              <Button kind="primary" size="sm" icon={<Plus size={16} />} onClick={startNewChat} className="mt-7">
                New chat
              </Button>
            </div>
          ) : filteredChats.length === 0 ? (
            // Laid out like the empty state; the cat is reading, glasses on, scanning for a match.
            <div className="flex flex-1 flex-col items-center justify-center pb-16 text-center animate-uzu-in">
              <div className="mb-1">
                <UzuCat state="reading" size={112} className="text-text-primary" label="uzu, searching" />
              </div>
              <div className="flex max-w-[320px] flex-col gap-1">
                <p className="text-[15px] font-[500] text-label-title">No matching chats</p>
                <p className="truncate text-[13px] text-label-muted">Nothing found for “{searchQuery.trim()}”</p>
              </div>
            </div>
          ) : (
            <div className="mt-4 mb-10 w-full flex flex-col gap-2 animate-uzu-in">
              {/* The list comes in once as a whole, like the settings page, not card by card. */}
              {filteredChats.map((chat) => (
                <div key={chat.id} className="relative">
                  <ChatCard
                    id={chat.id}
                    title={chat.title}
                    updatedAt={chat.updatedAt}
                    onClick={() => handleChatClick(chat.id)}
                    selected={isSelectionMode && selectedIds.includes(chat.id)}
                    selectMode={isSelectionMode}
                    checked={selectedIds.includes(chat.id)}
                    onCheckedChange={() => toggleSelectOne(chat.id)}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      </PageScroller>

      <ChatDeleteModal
        isOpen={isConfirmOpen}
        onClose={() => setIsConfirmOpen(false)}
        onConfirm={confirmBulkDelete}
        count={selectedIds.length}
        isDeleting={isDeleting}
      />

      {singleSelectedChat && (
        <ChatRenameModal
          isOpen={isRenameOpen}
          onClose={() => setIsRenameOpen(false)}
          onConfirm={async (newName) => {
            await useChatStore.getState().updateChatTitle(singleSelectedChat.id, newName);
            setIsRenameOpen(false);
          }}
          currentName={singleSelectedChat.title}
        />
      )}
    </div>
  );
}
