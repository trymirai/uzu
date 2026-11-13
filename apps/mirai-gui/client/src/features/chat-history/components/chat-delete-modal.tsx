import { Modal } from "@/components/ui/modal";

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
    ? `“${chatName}” will be deleted from this device. This can’t be undone.`
    : isBulk
      ? `${count} ${count === 1 ? "chat" : "chats"} will be deleted from this device. This can’t be undone.`
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
