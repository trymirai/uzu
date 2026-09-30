import type { MouseEvent, ReactNode } from "react";

export type ModelCardState =
  | { status: "available" }
  | { status: "downloading"; progress: number }
  | { status: "paused"; progress: number }
  | { status: "downloaded" }
  | { status: "error"; progress?: number };

export type ModelCardStatus = ModelCardState["status"];

export type ModelCardProps = {
  name: string;
  logo: ReactNode;
  parameters?: string;
  size?: string;
  state?: ModelCardState;
  onDownload?: () => void;
  onPause?: () => void;
  onCancel?: () => void;
  onDelete?: (event: MouseEvent<HTMLElement>) => void;
  quickDelete?: boolean;
  onRetry?: () => void;
  onOpen?: () => void;
  compact?: boolean;
};
