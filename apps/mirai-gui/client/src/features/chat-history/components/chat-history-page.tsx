import RenameIcon from "@/components/icons/rename-icon";
import TrashIcon from "@/components/icons/trash-icon";
import ChatCard from "./chat-card";
import { ChatDeleteModal } from "./chat-delete-modal";
import { ChatRenameModal } from "./chat-rename-modal";
import GlobalInstructions from "./global-instructions";
import { GroupButton } from "@/components/ui/group-button";
import { useNavigate } from "@tanstack/react-router";
import { X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { SearchInput } from "@/components/ui/search-input";
import { useToast } from "@/components/ui/toast/use-toast";
import { useChatStore } from "@/stores/use-chat-store";
import { useGlobalInstructionsStore } from "@/stores/use-global-instructions-store";

export function ChatHistoryPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const savedChats = useChatStore((s) => s.savedChats);
  const loadSavedChats = useChatStore((s) => s.loadSavedChats);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const instructions = useGlobalInstructionsStore((s) => s.instructions);
  const loadInstructions = useGlobalInstructionsStore((s) => s.loadInstructions);
  const saveInstructions = useGlobalInstructionsStore((s) => s.saveInstructions);

  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRenameOpen, setIsRenameOpen] = useState(false);

  useEffect(() => {
    loadSavedChats();
    loadInstructions();
  }, [loadSavedChats, loadInstructions]);

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
    <>
      <div className="sticky top-0 z-10">
        <div className="w-full bg-bg-modal">
          <div className="pt-7 pb-7 flex flex-col justify-center px-5 lg:px-0 lg:max-w-[800px] mx-auto gap-6 lg:gap-0">
            <h1 className="leading-[130%] text-xl font-medium text-label-title text-center lg:text-left">
              Chat history
            </h1>
          </div>

          <div className="px-5 lg:px-0 pt-4 pb-7 lg:max-w-[800px] mx-auto">
            <GlobalInstructions instructions={instructions} onSave={saveInstructions} />
          </div>
          <div className="h-[1px] bg-cell-border" />
        </div>

        <div className="w-full bg-background">
          <div className="px-5 lg:px-0 pt-5 pb-2.5 flex items-center justify-between lg:max-w-[800px] mx-auto">
            <h2 className="text-sm font-[350] text-label-title">Your chats</h2>
            <div className="flex items-center gap-2">
              <SearchInput
                value={searchQuery}
                onChange={setSearchQuery}
                placeholder="Search"
                className="w-full max-w-[210px] lg:min-w-[240px] lg:max-w-[240px]"
              />
              {isSelectionMode ? (
                <GroupButton>
                  <GroupButton.Segment ariaLabel="Select all" onClick={() => toggleSelectAll()} className="gap-2">
                    <Checkbox
                      checked={isAllSelected}
                      onChange={() => {}}
                      size="sm"
                      disabled
                      className="pointer-events-none opacity-100"
                    />
                    <span className={showActive ? "text-[12px] text-label-title" : "text-[12px] text-label-muted"}>
                      {labelText}
                    </span>
                  </GroupButton.Segment>

                  {!isAllSelected && singleSelectedChat && (
                    <GroupButton.Segment
                      leftDivider
                      ariaLabel="Rename"
                      onClick={() => setIsRenameOpen(true)}
                      className={showActive ? "text-label-title" : "text-label-muted"}
                    >
                      <RenameIcon className="w-4 h-4" />
                    </GroupButton.Segment>
                  )}

                  <GroupButton.Segment
                    leftDivider
                    ariaLabel="Delete"
                    onClick={() => setIsConfirmOpen(true)}
                    disabled={selectedIds.length === 0}
                    className={showActive ? "text-label-title" : "text-label-muted"}
                  >
                    <TrashIcon className="w-4 h-4" />
                  </GroupButton.Segment>

                  <GroupButton.Segment
                    leftDivider
                    ariaLabel="Exit selection"
                    onClick={exitSelectionMode}
                    className={showActive ? "text-label-title" : "text-label-muted"}
                  >
                    <X className="w-4 h-4" />
                  </GroupButton.Segment>
                </GroupButton>
              ) : (
                <Button
                  kind="secondary"
                  size="sm"
                  className="text-label-muted h-8 px-3 rounded-[8px] border border-cell-border text-sm leading-[150%]"
                  onClick={() => setIsSelectionMode(true)}
                >
                  Select
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="w-full lg:max-w-[808px] pl-2 overflow-auto mx-auto bg-background">
        <div className="px-5 lg:px-0 overflow-visible w-full">
          {filteredChats.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-label-muted">
                {searchQuery ? "No chats found matching your search." : "No chats yet. Start a new conversation!"}
              </p>
            </div>
          ) : (
            <div className="space-y-3 pb-6 w-full">
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
      </div>

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
    </>
  );
}
