import { Modal } from "@/components/ui/modal";
import { TextField } from "@/components/ui/text-field";
import { useEffect, useRef, useState } from "react";

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
  const wasOpen = useRef(false);

  useEffect(() => {
    if (isOpen && !wasOpen.current) {
      setName(currentName);
      setError("");
    }
    wasOpen.current = isOpen;
  }, [isOpen, currentName]);

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Name cannot be empty");
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
      initialFocus={inputRef}
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
            "aria-label": "Chat name",
            value: name,
            onChange: (e) => setName(e.target.value),
            // The input mounts after open; select on the focus the dialog gives it.
            onFocus: (e) => e.currentTarget.select(),
            placeholder: "Enter chat name",
            disabled: isSubmitting,
            fullWidth: true,
          }}
        />
      </form>
    </Modal>
  );
};
