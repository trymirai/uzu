import { CopyButton } from "@/components/ui/copy-button";
import { Button } from "@/components/ui/button";
import { TextArea } from "@/components/ui/text-area";
import { writeItemsWithFocus } from "@/utils/clipboard";
import { Pencil } from "lucide-react";
import { useId, useRef, useState } from "react";
import { attachmentStorage } from "../../services/attachment-storage";
import { AttachedFilesDisplay } from "../composer/attached-files-display";

export type UserMessageProps = {
  id: string;
  text: string;
  attachmentIds?: string[];
  canEdit?: boolean;
  onEdit?: (messageId: string, text: string, onSaved: () => void) => Promise<void>;
};

export const UserMessage = ({ id, text, attachmentIds, canEdit = false, onEdit }: UserMessageProps) => {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(text);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const saveInFlight = useRef(false);
  const descriptionId = useId();
  const editButtonRef = useRef<HTMLButtonElement>(null);
  const hasAttachments = (attachmentIds?.length ?? 0) > 0;
  const canSave = canEdit && !!onEdit && !saving && (draft.trim().length > 0 || hasAttachments);

  const copy = () =>
    writeItemsWithFocus([new ClipboardItem({ "text/plain": new Blob([text], { type: "text/plain" }) })]);

  const startEditing = () => {
    if (!canEdit || saveInFlight.current) return;
    setDraft(text);
    setError(null);
    setEditing(true);
  };

  const cancel = () => {
    if (saveInFlight.current) return;
    setEditing(false);
    requestAnimationFrame(() => editButtonRef.current?.focus());
  };

  const save = async () => {
    if (!canSave || !onEdit || saveInFlight.current) return;
    saveInFlight.current = true;
    setSaving(true);
    setError(null);
    let saved = false;
    try {
      await onEdit(id, draft, () => {
        saved = true;
        setEditing(false);
        setSaving(false);
      });
    } catch (cause) {
      if (!saved) setError(cause instanceof Error ? cause.message : "Could not save changes. Please try again.");
    } finally {
      saveInFlight.current = false;
      setSaving(false);
    }
  };

  return (
    <div className="group/user">
      {hasAttachments && (
        <div className="mb-3 ml-auto w-fit">
          <AttachedFilesDisplay files={attachmentStorage.getFiles(attachmentIds ?? [])} />
        </div>
      )}
      {editing ? (
        <div className="ml-auto flex w-full flex-col gap-3 rounded-lg bg-bg-hover p-3">
          <TextArea
            aria-label="Edit message"
            aria-describedby={descriptionId}
            autoFocus
            value={draft}
            onChange={setDraft}
            disabled={saving}
            maxHeightPx={320}
            className="bg-transparent text-[15px] leading-[140%]"
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) return;
              if (event.key === "Escape") {
                event.preventDefault();
                cancel();
              } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                void save();
              }
            }}
          />
          <p id={descriptionId} className="text-[12px] text-label-muted">
            Saving removes all later messages.
          </p>
          {error && (
            <p role="alert" className="text-[13px] text-error">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" size="sm" kind="ghost" disabled={saving} onClick={cancel}>
              Cancel
            </Button>
            <Button type="button" size="sm" disabled={!canSave} onClick={() => void save()}>
              {saving ? "Saving..." : "Save & send"}
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="font-[350] text-label-title relative w-fit max-w-full whitespace-pre-wrap [overflow-wrap:anywhere] rounded-[5px] text-[15px] leading-[140%] px-2.5 py-[6px] bg-bg-hover ml-auto">
            {text}
          </div>
          <div className="mt-1 flex h-8 items-center justify-end gap-1 opacity-0 transition-opacity group-hover/user:opacity-100 group-focus-within/user:opacity-100 [@media(hover:none)]:opacity-100">
            <CopyButton className="[&_svg]:size-4" onCopy={copy} />
            {onEdit && (
              <button
                ref={editButtonRef}
                type="button"
                aria-label="Edit message"
                title="Edit message"
                disabled={!canEdit}
                onClick={startEditing}
                className="flex min-h-8 min-w-8 items-center justify-center rounded-[5px] p-[6px] text-label-muted outline-hidden transition-colors hover:bg-card-hover hover:text-label-title focus-visible:shadow-focus disabled:opacity-50"
              >
                <Pencil className="size-4" />
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
};
