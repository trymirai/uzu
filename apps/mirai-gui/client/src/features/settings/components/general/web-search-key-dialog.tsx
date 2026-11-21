import { useRef, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { TextField } from "@/components/ui/text-field";
import {
  WEB_SEARCH_PROVIDERS,
  webSearchKeyOf,
  webSearchKeyTitle,
  type WebSearchKeyProvider,
} from "@/constants/web-search-providers";

type WebSearchKeyDialogProps = {
  open: boolean;
  provider: WebSearchKeyProvider;
  onClose: () => void;
  onSave: (key: string) => Promise<string | null>;
};

/** Each opening is a new dialog: the owner gives it a new `key`, so it starts empty. */
export function WebSearchKeyDialog({ open, provider, onClose, onSave }: WebSearchKeyDialogProps) {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { label } = WEB_SEARCH_PROVIDERS[provider];
  const title = webSearchKeyTitle(provider);
  const keyPage = webSearchKeyOf(provider).page;

  const save = async () => {
    if (saving || key.trim() === "") return;
    setSaving(true);
    const failure = await onSave(key);
    setSaving(false);
    if (failure) setError(failure);
    else onClose();
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      initialFocus={inputRef}
      title={title}
      description={`Searches go from this Mac to ${label} with your key. The key is stored in the macOS Keychain.`}
      primaryLabel={saving ? "Checking…" : "Save"}
      onPrimary={() => void save()}
      primaryDisabled={saving || key.trim() === ""}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void save();
        }}
      >
        <TextField
          error={error || undefined}
          controlProps={{
            ref: inputRef,
            type: "password",
            "aria-label": title,
            value: key,
            onChange: (event) => {
              setKey(event.target.value);
              setError("");
            },
            placeholder: "Paste your key",
            autoComplete: "off",
            spellCheck: false,
            disabled: saving,
            fullWidth: true,
          }}
        />
      </form>
      <p className="mt-2 text-[13px] leading-[1.45] text-text-secondary">
        Get a key at{" "}
        <a href={keyPage} target="_blank" rel="noreferrer" className="text-blue hover:underline">
          {keyPage.replace(/^https:\/\//, "").replace(/\/$/, "")}
        </a>
      </p>
    </Modal>
  );
}
