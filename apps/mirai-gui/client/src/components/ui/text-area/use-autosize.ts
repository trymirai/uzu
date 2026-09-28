import type { Ref } from "react";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { createAdjustTextAreaHeightHandler, setMergedRef } from "./utils";

type UseAutosizeParams = {
  ref: Ref<HTMLTextAreaElement> | undefined;
  value: string;
  rows: number;
  maxHeightPx?: number;
  enabled: boolean;
};

type UseAutosizeResult = {
  setRef: (node: HTMLTextAreaElement | null) => void;
  onInput: (() => void) | undefined;
};

export const useAutosize = ({ ref, value, rows, maxHeightPx, enabled }: UseAutosizeParams): UseAutosizeResult => {
  const localRef = useRef<HTMLTextAreaElement | null>(null);

  const setRef = useCallback(
    (node: HTMLTextAreaElement | null) => {
      localRef.current = node;
      setMergedRef(ref, node);
    },
    [ref],
  );

  const adjustHeight = useMemo(
    () =>
      createAdjustTextAreaHeightHandler({
        getTextarea: () => localRef.current,
        rows,
        maxHeightPx,
      }),
    [rows, maxHeightPx],
  );

  useEffect(() => {
    if (!enabled) return;
    adjustHeight();
  }, [value, enabled, adjustHeight]);

  return {
    setRef,
    onInput: enabled ? adjustHeight : undefined,
  };
};
