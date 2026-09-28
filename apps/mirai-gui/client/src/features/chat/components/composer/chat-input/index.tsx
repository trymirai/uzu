import { Settings } from "lucide-react";
import { useCallback, useRef } from "react";
import { twMerge } from "tailwind-merge";
import { TextArea } from "@/components/ui/text-area";
import { AttachMenu } from "./attach-menu";
import { FileChip } from "./file-chip";
import { ModelPicker } from "./model-picker";
import { SendButton } from "./send-button";
import { CHAT_INPUT_TEXTAREA_CLASSNAME, CHAT_INPUT_TEXTAREA_STYLE } from "./constants";
import type { ChatInputProps } from "./types";
import { useChatInputController } from "./use-chat-input-controller";

export function ChatInput({
  value,
  onChange,
  onSend,
  canSend,
  onBlockedSend,
  onAttach,
  attachAccept = ".txt,.md,.json,.csv,.yaml,.yml",
  placeholder = "Send a message to the local model…",
  files,
  onRemoveFile,
  models,
  activeModelId,
  onModelChange,
  onModelSettingsClick,
  moreModelsLink,
  modelPickerDisabled = false,
  streaming = false,
  onStop,
  settingsModified = false,
  settingsDisabled = false,
}: ChatInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hasFiles = files && files.length > 0;

  const controller = useChatInputController({
    value,
    onChange,
    files,
    onSend,
    canSend,
    onBlockedSend,
  });

  const handleFileSelect = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="rounded-xl bg-surface-elevated shadow-md p-4 w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept={attachAccept}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            void Promise.resolve(onAttach?.(file));
          }
          e.target.value = "";
        }}
      />

      {hasFiles && (
        <div className="flex flex-wrap gap-2 mb-3">
          {files.map((file) => (
            <FileChip key={file.id} file={file} onRemove={() => onRemoveFile?.(file.id)} />
          ))}
        </div>
      )}

      <TextArea
        ref={controller.textareaRef}
        value={value}
        onChange={onChange}
        onKeyDown={controller.handleKeyDown}
        placeholder={placeholder}
        rows={1}
        maxHeightPx={320}
        className={twMerge(CHAT_INPUT_TEXTAREA_CLASSNAME, "bg-transparent")}
        style={{ ...CHAT_INPUT_TEXTAREA_STYLE, minHeight: 23, maxHeight: 320 }}
      />

      <div className="flex items-center justify-between mt-4">
        <AttachMenu onSelectFile={handleFileSelect} />

        <div className="flex items-center gap-2 min-w-0">
          <ModelPicker
            models={models}
            {...(activeModelId ? { activeModelId } : {})}
            onModelChange={onModelChange}
            moreModelsLink={moreModelsLink}
            disabled={modelPickerDisabled}
          />
          {onModelSettingsClick ? (
            <span className="relative inline-flex shrink-0">
              <button
                type="button"
                onClick={onModelSettingsClick}
                disabled={settingsDisabled}
                aria-label="Model settings"
                className="flex size-7 shrink-0 items-center justify-center rounded-md bg-surface-tertiary text-text-muted outline-hidden transition-colors duration-150 ease-out hover:bg-control-surface-hover hover:text-text-primary focus-visible:shadow-focus disabled:pointer-events-none disabled:opacity-50"
              >
                <Settings size={16} />
              </button>
              {settingsModified && (
                <span className="pointer-events-none absolute right-0.5 top-0.5 size-1.5 rounded-full bg-danger ring-1 ring-surface-elevated" />
              )}
            </span>
          ) : null}
          <SendButton
            disabled={controller.sendDisabled}
            hasContent={controller.hasContent}
            onClick={controller.handleSubmit}
            streaming={streaming}
            onStop={onStop}
          />
        </div>
      </div>
    </div>
  );
}
