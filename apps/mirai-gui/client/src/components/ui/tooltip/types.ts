import type { ReactNode } from "react";

export type TooltipSide = "top" | "bottom";

export type TooltipProviderProps = {
  children: ReactNode;
};

export type TooltipProps = {
  content: ReactNode;
  side?: TooltipSide;
  /** How long the pointer rests before the first tooltip opens. Defaults to the shared 400ms; a tip
   *  that carries real content (not a label) can open at once, like a hover menu. */
  delay?: number;
  /** Centred on the trigger by default; "start" lines its edge up with the trigger's, like a menu. */
  align?: "center" | "start";
  /** "label" is the small text tip. "panel" carries real content and takes the hover menus' shape:
   *  the same corners, inset and distance from the trigger, so the two read as one family. */
  variant?: "label" | "panel";
  children: ReactNode;
};

export type TooltipPayload = {
  id: string;
  reference: HTMLSpanElement;
  content: TooltipProps["content"];
  side: NonNullable<TooltipProps["side"]>;
  delay?: number;
  align?: TooltipProps["align"];
  variant?: TooltipProps["variant"];
};

export type TooltipContextValue = {
  activeId: string | null;
  tooltipId: string;
  show: (payload: TooltipPayload) => void;
  hide: (id: string) => void;
};
