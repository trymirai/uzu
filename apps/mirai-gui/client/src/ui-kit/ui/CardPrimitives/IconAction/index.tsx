import type { MouseEvent, ReactNode } from "react";
import { Button } from "../../Button";

type IconActionProps = {
  label: string;
  icon: ReactNode;
  onClick?: () => void;
  className?: string;
};

export function IconAction({ label, icon, onClick, className }: IconActionProps) {
  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    onClick?.();
  };

  return (
    <Button
      size="xs"
      kind="ghost"
      iconOnly
      ariaLabel={label}
      icon={icon}
      onClick={handleClick}
      className={
        className ??
        "size-7 rounded-md bg-transparent text-text-muted hover:bg-border-default hover:text-text-primary active:bg-border-default"
      }
    />
  );
}
