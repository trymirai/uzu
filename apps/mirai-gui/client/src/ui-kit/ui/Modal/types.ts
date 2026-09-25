import type { ReactNode } from "react";
import type { ButtonKind } from "../Button/types";

export type ModalProps = {
  open: boolean;
  onClose: () => void;

  title: string;
  description?: string;
  children?: ReactNode;

  primaryLabel?: string;
  primaryKind?: ButtonKind;
  onPrimary?: () => void;
  primaryDisabled?: boolean;

  secondaryLabel?: string;
  onSecondary?: () => void;
};
