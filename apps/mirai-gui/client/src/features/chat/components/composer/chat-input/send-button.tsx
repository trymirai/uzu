import { twMerge } from "tailwind-merge";
import { Button } from "@/components/ui/button";
import { IconPlayerStopFilled } from "@/components/icons/player-stop-filled-icon";
import { IconSend } from "@/components/icons/send-icon";

export type SendButtonProps = {
  disabled: boolean;
  hasContent: boolean;
  onClick: () => void;
  streaming?: boolean;
  onStop?: () => void;
};

export function SendButton({ disabled, hasContent, onClick, streaming = false, onStop }: SendButtonProps) {
  if (streaming) {
    return (
      <Button
        size="xs"
        kind="primary"
        iconOnly
        ariaLabel="Stop generating"
        icon={
          <span key="stop" className="flex animate-uzu-swap">
            <IconPlayerStopFilled size={14} />
          </span>
        }
        onClick={onStop}
        className="rounded-full active:scale-100"
      />
    );
  }

  return (
    <Button
      size="xs"
      kind={hasContent ? "primary" : "ghost"}
      iconOnly
      ariaLabel="Send message"
      icon={
        <span key="send" className="flex animate-uzu-swap">
          <IconSend />
        </span>
      }
      disabled={disabled}
      onClick={onClick}
      className={twMerge(
        "rounded-full active:scale-100",
        hasContent ? "" : "bg-control-fill text-text-disabled cursor-not-allowed opacity-100",
      )}
    />
  );
}
