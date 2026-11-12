import { SettingsIcon } from "@/components/left-bar/nav-icons";
import { Button } from "@headlessui/react";
import { useCallback, useRef } from "react";
import { twMerge } from "tailwind-merge";
import { TextArea } from "@/components/ui/text-area";
import { AttachMenu } from "./attach-menu";
import { FileChip } from "./file-chip";
import { ModelPicker } from "./model-picker";
import { ReasoningSelector } from "./reasoning-selector";
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
  placeholder = "Send a message to the model…",
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
    files,
    onSend,
    canSend,
    onBlockedSend,
  });

  const handleFileSelect = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="rounded-[20px] bg-surface-elevated shadow-md px-4 pt-4 pb-3 w-full transition-shadow duration-150 ease-out focus-within:shadow-lg dark:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.1)] dark:focus-within:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.18)]">
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
        data-composer-input
        value={value}
        onChange={onChange}
        onKeyDown={controller.handleKeyDown}
        placeholder={placeholder}
        rows={1}
        maxHeightPx={320}
        className={twMerge(CHAT_INPUT_TEXTAREA_CLASSNAME, "bg-transparent")}
        style={{ ...CHAT_INPUT_TEXTAREA_STYLE, minHeight: 23, maxHeight: 320 }}
      />

      <div className="flex items-center justify-between gap-2 mt-4">
        <AttachMenu onSelectFile={handleFileSelect} />

        <div className="flex items-center gap-2 min-w-0">
          <ModelPicker
            models={models}
            {...(activeModelId ? { activeModelId } : {})}
            onModelChange={onModelChange}
            moreModelsLink={moreModelsLink}
            disabled={modelPickerDisabled}
          />
          {activeModelId ? (
            <ReasoningSelector key={activeModelId} repoId={activeModelId} disabled={modelPickerDisabled} />
          ) : null}
          {onModelSettingsClick ? (
            <span className="relative inline-flex shrink-0">
              <Button
                type="button"
                onClick={onModelSettingsClick}
                disabled={settingsDisabled}
                aria-label="Model settings"
                className="flex size-7 shrink-0 items-center justify-center rounded-full bg-control-fill text-text-secondary outline-hidden transition-colors duration-150 ease-out hover:bg-control-fill-hover hover:text-text-primary data-[focus]:shadow-focus disabled:pointer-events-none disabled:opacity-50"
              >
                {/* Lands half a point low on Retina screens; lifted back to the circle's centre. */}
                <SettingsIcon size={16} aria-hidden />
              </Button>
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
