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
        icon={<IconPlayerStopFilled size={14} />}
        onClick={onStop}
        className="active:scale-100"
      />
    );
  }

  return (
    <Button
      size="xs"
      kind={hasContent ? "primary" : "ghost"}
      iconOnly
      ariaLabel="Send message"
      icon={<IconSend />}
      disabled={disabled}
      onClick={onClick}
      className={twMerge(
        "active:scale-100",
        hasContent ? "" : "bg-surface-tertiary text-text-disabled cursor-not-allowed opacity-100",
      )}
    />
  );
}
