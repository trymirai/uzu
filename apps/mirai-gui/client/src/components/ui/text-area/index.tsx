import { forwardRef } from "react";
import { twMerge } from "tailwind-merge";
import { useAutosize } from "./use-autosize";
import type { TextAreaProps } from "./types";

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(function TextArea(
  { value, onChange, rows = 3, maxHeightPx, className = "", ...rest }: TextAreaProps,
  ref,
) {
  const resolvedValue = value ?? "";
  const { setRef, onInput } = useAutosize({
    ref,
    value: resolvedValue,
    rows,
    enabled: maxHeightPx !== undefined,
    maxHeightPx,
  });

  return (
    <textarea
      ref={setRef}
      className={twMerge(
        "w-full resize-none outline-hidden overscroll-y-auto thin-scrollbar text-text-primary placeholder:text-text-muted",
        className,
      )}
      value={resolvedValue}
      rows={rows}
      onChange={(e) => onChange(e.target.value)}
      onInput={onInput}
      {...rest}
    />
  );
});
