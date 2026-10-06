import type { ReactNode } from "react";

export type TooltipSide = "top" | "bottom";

export type TooltipProviderProps = {
  children: ReactNode;
};

export type TooltipProps = {
  content: ReactNode;
  side?: TooltipSide;
  children: ReactNode;
};

export type TooltipPayload = {
  id: string;
  reference: HTMLSpanElement;
  content: TooltipProps["content"];
  side: NonNullable<TooltipProps["side"]>;
};

export type TooltipContextValue = {
  activeId: string | null;
  tooltipId: string;
  show: (payload: TooltipPayload) => void;
  hide: (id: string) => void;
};
