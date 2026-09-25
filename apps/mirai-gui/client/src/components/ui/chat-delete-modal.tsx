import { Modal } from "@/ui-kit";

type ChatDeleteModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  chatName?: string;
  count?: number;
  isDeleting?: boolean;
};

export const ChatDeleteModal: React.FC<ChatDeleteModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  chatName,
  count,
  isDeleting = false,
}) => {
  const isBulk = typeof count === "number" && count > 0 && !chatName;
  const description = chatName
    ? `Are you sure you want to delete chat "${chatName}"? This action cannot be undone.`
    : isBulk
      ? `Are you sure you want to delete ${count} ${count === 1 ? "chat" : "chats"}? This action cannot be undone.`
      : "";

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title={chatName ? "Delete chat" : "Delete chats"}
      description={description}
      primaryLabel={chatName ? "Delete chat" : "Delete"}
      primaryKind="danger"
      onPrimary={onConfirm}
      primaryDisabled={isDeleting}
    />
  );
};
