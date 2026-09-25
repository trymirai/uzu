import type { MouseEvent, ReactElement, ReactNode } from "react";

export type PopoverItem = {
  id: string;
  label: ReactNode;
  icon?: ReactNode;
  danger?: boolean;
  disabled?: boolean;
  onClick?: (event: MouseEvent<HTMLElement>) => void;
};

export type PopoverSide = "top" | "bottom";
export type PopoverAlign = "start" | "end";

export type PopoverTrigger = ReactElement | ((open: boolean) => ReactElement);
