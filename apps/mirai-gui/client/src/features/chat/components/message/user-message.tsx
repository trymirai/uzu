import { CopyButton } from "@/components/ui/copy-button";
import { Button as HeadlessButton } from "@headlessui/react";
import { TextArea } from "@/components/ui/text-area";
import { writeItemsWithFocus } from "@/utils/clipboard";
import { PencilIcon } from "@/components/icons/app-icons";
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
    <div className="group/user relative">
      {hasAttachments && (
        <div className="mb-3 ml-auto w-fit">
          <AttachedFilesDisplay files={attachmentStorage.getFiles(attachmentIds ?? [])} />
        </div>
      )}
      {editing ? (
        // Drawn like the message box below, since that's what it becomes for a moment: the same
        // card, the text in the same place, and its actions in one row along the bottom.
        <div className="ml-auto flex w-full flex-col rounded-[20px] bg-surface-elevated px-4 pt-4 pb-3 shadow-md transition-shadow duration-150 ease-out focus-within:shadow-lg dark:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.1)] dark:focus-within:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.18)]">
          <TextArea
            aria-label="Edit message"
            aria-describedby={descriptionId}
            autoFocus
            // Opens with the caret after the text, ready to add to it.
            onFocus={(event) => {
              const end = event.currentTarget.value.length;
              event.currentTarget.setSelectionRange(end, end);
            }}
            value={draft}
            onChange={setDraft}
            disabled={saving}
            rows={1}
            maxHeightPx={320}
            className="bg-transparent text-[15px] leading-[1.5]"
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
          {error && (
            <p role="alert" className="mt-2 text-[13px] text-error">
              {error}
            </p>
          )}
          <div className="mt-4 flex items-center justify-between gap-3">
            <p id={descriptionId} className="min-w-0 text-[12px] leading-[1.4] text-label-muted text-pretty">
              Saving removes all later messages.
            </p>
            <div className="flex shrink-0 items-center gap-1.5">
              <button
                type="button"
                disabled={saving}
                onClick={cancel}
                className="h-7 rounded-full px-3 text-[13px] font-[450] text-text-secondary outline-hidden transition-colors duration-150 ease-out hover:bg-control-fill hover:text-text-primary focus-visible:shadow-focus disabled:pointer-events-none disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!canSave}
                onClick={() => void save()}
                className="h-7 rounded-full bg-primary px-3 text-[13px] font-[500] text-primary-contrast outline-hidden transition-[background-color,scale] duration-150 ease-out hover:bg-primary-hover active:scale-[0.97] focus-visible:shadow-focus disabled:pointer-events-none disabled:opacity-40"
              >
                {saving ? "Saving…" : "Save & send"}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="font-[400] text-user-bubble-text relative w-fit max-w-full whitespace-pre-wrap [overflow-wrap:anywhere] rounded-[18px] text-[15px] leading-[1.5] px-3.5 py-2 bg-user-bubble ml-auto">
            {text}
          </div>
          {/* Floats under the bubble on hover rather than holding an empty row open below it. */}
          <div className="absolute right-0 top-full z-10 mt-1 flex h-8 items-center justify-end gap-1 opacity-0 transition-opacity group-hover/user:opacity-100 group-focus-within/user:opacity-100 [@media(hover:none)]:opacity-100">
            <CopyButton className="[&_svg]:size-4" onCopy={copy} />
            {onEdit && (
              <HeadlessButton
                ref={editButtonRef}
                type="button"
                aria-label="Edit message"
                title="Edit message"
                disabled={!canEdit}
                onClick={startEditing}
                className="flex min-h-8 min-w-8 items-center justify-center rounded-lg p-[6px] text-label-muted outline-hidden transition-colors hover:bg-card-hover hover:text-label-title data-[focus]:shadow-focus disabled:pointer-events-none disabled:opacity-50"
              >
                <PencilIcon size={16} />
              </HeadlessButton>
            )}
          </div>
        </>
      )}
    </div>
  );
};
