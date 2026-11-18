import { useEffect, useRef, useState } from "react";
import { UzuCat, UzuCatStill, type UzuCatState } from "@/components/uzu-cat";
import { TextShimmer } from "@/components/ui/text-shimmer";

/** What the run behind an assistant message is doing right now. */
export type AssistantPhase =
  "loadingModel" | "reading" | "thinking" | "tool" | "writing" | "done" | "stopped" | "error" | "outOfMemory";

const CAT: Record<AssistantPhase, UzuCatState> = {
  loadingModel: "loading",
  reading: "reading",
  thinking: "thinking",
  tool: "tool",
  writing: "writing",
  done: "done",
  stopped: "idle",
  error: "idle",
  outOfMemory: "memory",
};

/** Before the first token, the cat's row carries the status in words too. */
export const PENDING_LABEL: Partial<Record<AssistantPhase, string>> = {
  loadingModel: "Loading the model…",
  reading: "Reading your message…",
};

export const CAT_SIZE = 28;

/**
 * The cat beside an answer. The newest answer gets the live cat, which follows the run (goggles
 * while the prompt is read, a daydream while thinking, gears for tools, lines of text while
 * writing), is pleased for a moment when it finishes, and then looks at you. Older answers keep a
 * still cat so the history stays calm.
 */
export function AssistantCat({ phase, live }: { phase: AssistantPhase; live: boolean }) {
  // Pleased only when an answer finishes in front of you, not when an old chat is opened.
  const [cheering, setCheering] = useState(false);
  const previous = useRef(phase);
  useEffect(() => {
    const finished = phase === "done" && previous.current !== "done";
    previous.current = phase;
    if (!finished) return;
    setCheering(true);
    const timer = setTimeout(() => setCheering(false), 1800);
    return () => clearTimeout(timer);
  }, [phase]);

  if (!live) return <UzuCatStill size={CAT_SIZE} className="text-text-primary" />;
  const state: UzuCatState = phase === "done" ? (cheering ? "done" : "follow") : CAT[phase];
  return (
    <UzuCat state={state} size={CAT_SIZE} className="text-text-primary" label={`uzu, ${LABEL[state] ?? "ready"}`} />
  );
}

const LABEL: Partial<Record<UzuCatState, string>> = {
  loading: "loading the model",
  reading: "reading",
  thinking: "thinking",
  tool: "using a tool",
  writing: "writing",
  done: "done",
  memory: "short on memory",
};

/** The status row shown before an answer has any content. */
export function PendingStatus({ phase }: { phase: AssistantPhase }) {
  const label = PENDING_LABEL[phase];
  if (!label) return null;
  return (
    <span key={phase} className="flex h-7 items-center animate-uzu-swap">
      <TextShimmer
        duration={1.6}
        className="text-[15px] font-[400] leading-[1.3] [--base-color:var(--color-label-muted)]"
      >
        {label}
      </TextShimmer>
    </span>
  );
}
