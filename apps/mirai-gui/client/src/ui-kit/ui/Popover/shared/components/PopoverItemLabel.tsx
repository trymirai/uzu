import type { ReactNode } from "react";
import { TEXT_13, TEXT_13_STYLE } from "../constants";
import { isPlainTextLabel } from "../utils";

type PopoverItemLabelProps = {
  label: ReactNode;
};

export function PopoverItemLabel({ label }: PopoverItemLabelProps) {
  if (!isPlainTextLabel(label)) {
    return <>{label}</>;
  }

  return (
    <span className={TEXT_13} style={TEXT_13_STYLE}>
      {label}
    </span>
  );
}
