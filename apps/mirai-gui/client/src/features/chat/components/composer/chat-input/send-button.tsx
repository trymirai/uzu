import { twMerge } from "tailwind-merge";
import { Button } from "@/components/ui/button";
import { SquareIcon } from "@/components/icons/square-icon";
import { ArrowUpIcon } from "@/components/icons/arrow-up-icon";

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
            <SquareIcon size={14} />
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
          <ArrowUpIcon />
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
