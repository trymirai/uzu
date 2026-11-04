import { Textarea as HeadlessTextarea } from "@headlessui/react";
import { Plus } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useToast } from "@/components/ui/toast/use-toast";

type GlobalInstructionsProps = {
  instructions: string;
  onSave: (instructions: string) => Promise<boolean>;
};

const GlobalInstructions: React.FC<GlobalInstructionsProps> = ({ instructions, onSave }) => {
  const toast = useToast();
  const [isExpanded, setIsExpanded] = useState(false);
  const [localInstructions, setLocalInstructions] = useState(instructions);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  // null after a failed save, so the next blur saves again.
  const lastRequestedRef = useRef<string | null>(instructions || "");
  const latestValueRef = useRef<string>(instructions || "");
  const touchedRef = useRef(false);

  // After the first keystroke the store only repeats what this editor saved.
  useEffect(() => {
    if (touchedRef.current) return;
    setLocalInstructions(instructions);
    lastRequestedRef.current = instructions || "";
  }, [instructions]);

  useEffect(() => {
    latestValueRef.current = localInstructions;
  }, [localInstructions]);

  const doSave = useCallback(
    (value: string) => {
      if (value === lastRequestedRef.current) return;
      lastRequestedRef.current = value;
      void onSave(value).then((saved) => {
        if (saved) return;
        if (lastRequestedRef.current === value) lastRequestedRef.current = null;
        toast.error("Failed to save instructions", { id: "instructions-save-failed" });
      });
    },
    [onSave, toast],
  );

  const scheduleSave = useCallback(
    (value: string) => {
      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
      saveTimerRef.current = window.setTimeout(() => {
        doSave(value);
        saveTimerRef.current = null;
      }, 500);
    },
    [doSave],
  );

  const flushSave = useCallback(() => {
    if (saveTimerRef.current) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    const value = latestValueRef.current;
    doSave(value);
  }, [doSave]);

  useEffect(() => {
    const handleBeforeUnload = () => flushSave();
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      flushSave();
    };
  }, [flushSave]);

  useEffect(() => {
    if (!isExpanded) return;
    const id = requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      const len = el.value.length;
      el.setSelectionRange(len, len);
      el.scrollTop = el.scrollHeight;
    });
    return () => cancelAnimationFrame(id);
  }, [isExpanded]);

  const handleToggleExpand = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value;
      touchedRef.current = true;
      setLocalInstructions(value);
      scheduleSave(value);
    },
    [scheduleSave],
  );

  return (
    <div className="rounded-xl bg-surface-elevated shadow-md dark:shadow-[inset_0_0_0_0.5px_rgb(255_255_255/0.1)]">
      <button
        type="button"
        aria-expanded={isExpanded}
        className="group flex min-h-12 w-full items-center gap-3 rounded-xl pl-3 pr-4 py-2.5 text-left outline-hidden focus-visible:shadow-focus"
        onClick={handleToggleExpand}
      >
        <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-tertiary text-label-muted transition-colors duration-150 ease-out group-hover:bg-control-surface-hover group-hover:text-label-title">
          <Plus
            aria-hidden
            className={`size-4 transition-transform duration-200 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none ${isExpanded ? "rotate-45" : ""}`}
          />
        </span>
        <span className="flex-1 min-w-0 text-[14px] font-[450] leading-[1.3] text-label-title">
          Add instructions to all chats
        </span>
        <span className="hidden sm:block text-[13px] leading-[1.3] text-label-muted">
          Tailor the way the model responds
        </span>
      </button>

      {/* Opens in place: the card grows from its own row instead of a second box appearing. */}
      <div
        className={`grid transition-[grid-template-rows,opacity] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none ${isExpanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
      >
        <div className="min-h-0 overflow-clip">
          <div className="px-4 pb-4">
            <HeadlessTextarea
              ref={textareaRef}
              value={localInstructions}
              onChange={handleChange}
              onBlur={flushSave}
              placeholder="Add instructions to all chats"
              tabIndex={isExpanded ? 0 : -1}
              className="w-full rounded-xl bg-surface-soft px-3 py-2.5 text-[14px] leading-[1.5] text-label-title placeholder:text-label-muted focus:outline-hidden resize-none overscroll-y-auto thin-scrollbar"
              rows={6}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default GlobalInstructions;
