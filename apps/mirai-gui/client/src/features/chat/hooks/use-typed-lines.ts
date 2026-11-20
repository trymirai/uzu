import { useEffect, useState } from "react";

const STEP_MS = 24;
const LINE_MAX_MS = 420;

// A query typed once, in the card or in a search row, is shown whole from then on.
const typedLines = new Set<string>();

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

type Progress = { line: number; chars: number };

export function useTypedLines(lines: readonly string[]) {
  const [progress, setProgress] = useState<Progress>(() => {
    let line = 0;
    while (line < lines.length && typedLines.has(lines[line]!)) line += 1;
    return { line, chars: 0 };
  });
  // Replies re-render on every streamed token; only a change in the lines restarts the timer.
  const linesKey = lines.join("\u0000");

  useEffect(() => {
    const current = lines[progress.line];
    if (current === undefined) return;
    if (progress.chars >= current.length || typedLines.has(current) || prefersReducedMotion()) {
      typedLines.add(current);
      setProgress({ line: progress.line + 1, chars: 0 });
      return;
    }
    const timer = setTimeout(
      () => setProgress({ line: progress.line, chars: progress.chars + 1 }),
      Math.min(STEP_MS, LINE_MAX_MS / current.length),
    );
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `linesKey` stands for `lines`
  }, [linesKey, progress]);

  return {
    shown: Math.min(lines.length, progress.line + 1),
    textOf: (index: number): string =>
      index < progress.line ? lines[index]! : index === progress.line ? lines[index]!.slice(0, progress.chars) : "",
    isTyping: (index: number): boolean => index === progress.line && index < lines.length,
    isTyped: (index: number): boolean => index < progress.line,
    allTyped: progress.line >= lines.length,
  };
}
