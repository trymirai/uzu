import { Textarea as HeadlessTextarea } from "@headlessui/react";
import { Plus } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { CardContainer } from "@/components/ui/card-container";
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
    <div>
      <CardContainer>
        <button
          className="w-full group flex items-center gap-3 p-3 text-left lg:justify-between"
          onClick={handleToggleExpand}
        >
          <div className="flex items-center gap-3 flex-1 min-w-0 lg:flex-initial">
            <Plus
              className={`w-5 h-5 shrink-0 text-label-muted transition-transform duration-200 ${isExpanded ? "rotate-45" : ""}`}
            />
            <div className="flex-1 min-w-0 lg:flex-initial">
              <h3 className="text-sm font-[350] leading-[150%] text-label-title">Add instructions to all chats</h3>
              <span className="block text-[13px] font-[350] leading-[130%] text-label-muted lg:hidden">
                Tailor the way the model responds
              </span>
            </div>
          </div>

          <span className="hidden lg:block text-[13px] font-[350] leading-[130%] text-label-muted text-right">
            Tailor the way the model responds
          </span>
        </button>
      </CardContainer>

      <CardContainer
        className={`overflow-hidden transition-all duration-300 ease-in-out ${isExpanded ? "max-h-96 opacity-100 mt-3" : "max-h-0 opacity-0"}`}
      >
        <div className="p-3">
          <HeadlessTextarea
            ref={textareaRef}
            value={localInstructions}
            onChange={handleChange}
            onBlur={flushSave}
            placeholder="Add instructions to all chats"
            className="w-full p-1 bg-bg-modal rounded-md text-sm leading-[150%] text-label-title placeholder:text-label-muted focus:outline-hidden focus:border-primary resize-none thin-scrollbar"
            rows={7}
          />
        </div>
      </CardContainer>
    </div>
  );
};

export default GlobalInstructions;
