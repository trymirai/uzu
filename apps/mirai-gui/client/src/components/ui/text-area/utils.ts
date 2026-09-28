import type { MutableRefObject, Ref } from "react";

type TextAreaResizeParams = {
  textarea: HTMLTextAreaElement;
  maxHeightPx?: number;
  rows: number;
};

type TextAreaAdjustOptions = {
  getTextarea: () => HTMLTextAreaElement | null;
  rows: number;
  maxHeightPx?: number;
};

const isMutableRef = (
  value: Ref<HTMLTextAreaElement> | undefined,
): value is MutableRefObject<HTMLTextAreaElement | null> =>
  value !== null && typeof value === "object" && "current" in value;

export const setMergedRef = (ref: Ref<HTMLTextAreaElement> | undefined, node: HTMLTextAreaElement | null) => {
  if (typeof ref === "function") {
    ref(node);
    return;
  }
  if (isMutableRef(ref)) {
    ref.current = node;
  }
};

const adjustTextAreaHeight = ({ textarea, maxHeightPx, rows }: TextAreaResizeParams) => {
  textarea.style.height = "auto";
  const styles = window.getComputedStyle(textarea);
  const lineHeight = parseInt(styles.lineHeight, 10);
  const paddingTop = parseInt(styles.paddingTop, 10);
  const paddingBottom = parseInt(styles.paddingBottom, 10);
  const minHeightFromStyle = parseInt(styles.minHeight, 10);
  const scrollHeight = textarea.scrollHeight;
  const minHeightComputed = lineHeight * rows + paddingTop + paddingBottom;
  const minHeight = Number.isNaN(minHeightFromStyle)
    ? minHeightComputed
    : Math.max(minHeightComputed, minHeightFromStyle);
  const nextHeight = Math.max(scrollHeight, minHeight);

  if (maxHeightPx !== undefined && nextHeight > maxHeightPx) {
    textarea.style.height = `${maxHeightPx}px`;
    textarea.style.overflowY = "auto";
    return;
  }
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = "hidden";
};

export const createAdjustTextAreaHeightHandler = ({ getTextarea, rows, maxHeightPx }: TextAreaAdjustOptions) => {
  return () => {
    const textarea = getTextarea();
    if (!textarea) return;
    adjustTextAreaHeight({ textarea, rows, maxHeightPx });
  };
};
