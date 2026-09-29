import { Modal } from "@/components/ui/modal";
import { TextField } from "@/components/ui/text-field";
import { useEffect, useRef, useState } from "react";
import { CHAT_TITLE_MAX_LENGTH, CHAT_TITLE_MIN_LENGTH } from "@/constants/chat";

type ChatRenameModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (name: string) => Promise<void>;
  currentName: string;
};

export const ChatRenameModal: React.FC<ChatRenameModalProps> = ({ isOpen, onClose, onConfirm, currentName }) => {
  const [name, setName] = useState(currentName);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setName(currentName);
      setError("");
      setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 100);
    }
  }, [isOpen, currentName]);

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Name cannot be empty");
      return;
    }
    if (trimmed.length < CHAT_TITLE_MIN_LENGTH) {
      setError(`Name must be at least ${CHAT_TITLE_MIN_LENGTH} characters long`);
      return;
    }
    if (trimmed.length > CHAT_TITLE_MAX_LENGTH) {
      setError(`Name must be at most ${CHAT_TITLE_MAX_LENGTH} characters long`);
      return;
    }
    try {
      setIsSubmitting(true);
      await onConfirm(trimmed);
      onClose();
    } catch (e) {
      console.error(e);
      setError("Failed to rename chat");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setName(currentName);
    setError("");
    onClose();
  };

  return (
    <Modal
      open={isOpen}
      onClose={handleClose}
      title="Rename chat"
      primaryLabel={isSubmitting ? "Saving…" : "Save"}
      onPrimary={() => void submit()}
      primaryDisabled={isSubmitting}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <TextField
          error={error || undefined}
          controlProps={{
            ref: inputRef,
            value: name,
            onChange: (e) => setName(e.target.value),
            placeholder: "Enter chat name",
            maxLength: CHAT_TITLE_MAX_LENGTH,
            disabled: isSubmitting,
            fullWidth: true,
          }}
        />
      </form>
    </Modal>
  );
};
