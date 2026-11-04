import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { UzuCat } from "@/components/uzu-cat";

const COMPOSER_SELECTOR = "textarea[data-composer-input]";

/** Where the caret of a textarea is on screen, measured with an invisible copy of its text. */
function caretPoint(textarea: HTMLTextAreaElement): { x: number; y: number } {
  const style = window.getComputedStyle(textarea);
  const mirror = document.createElement("div");
  for (const prop of [
    "boxSizing",
    "width",
    "paddingTop",
    "paddingRight",
    "paddingBottom",
    "paddingLeft",
    "borderTopWidth",
    "borderRightWidth",
    "borderBottomWidth",
    "borderLeftWidth",
    "fontFamily",
    "fontSize",
    "fontWeight",
    "fontVariationSettings",
    "lineHeight",
    "letterSpacing",
    "wordSpacing",
    "tabSize",
  ] as const) {
    mirror.style[prop] = style[prop];
  }
  Object.assign(mirror.style, {
    position: "fixed",
    top: "0",
    left: "-9999px",
    visibility: "hidden",
    whiteSpace: "pre-wrap",
    overflowWrap: "break-word",
  });
  mirror.textContent = textarea.value.slice(0, textarea.selectionEnd ?? textarea.value.length);
  const marker = document.createElement("span");
  marker.textContent = "​";
  mirror.appendChild(marker);
  document.body.appendChild(mirror);
  const box = textarea.getBoundingClientRect();
  const x = box.left + marker.offsetLeft - textarea.scrollLeft;
  const y = box.top + marker.offsetTop - textarea.scrollTop + marker.offsetHeight / 2;
  mirror.remove();
  return { x, y };
}

/** Where to look, -1..1 each way, from the cat toward a target on screen. Scaled so something a few
 *  hundred pixels away already reads as "over there". */
function gazeToward(target: DOMRect, face: DOMRect): { x: number; y: number } {
  const clamp = (v: number) => Math.max(-1, Math.min(1, v));
  return {
    x: clamp((target.left + target.width / 2 - (face.left + face.width / 2)) / 320),
    y: clamp((target.top + target.height / 2 - (face.top + face.height / 2)) / 160),
  };
}

/**
 * The empty state of a new chat: the cat waits for you, glancing at the box, and while you type it
 * watches the caret move along the line.
 */
export function NewChatCat({ draft, missingModelAt = 0 }: { draft: string; missingModelAt?: number }) {
  const catRef = useRef<HTMLSpanElement>(null);
  const [lookAt, setLookAt] = useState<{ x: number; y: number } | null>(null);

  // Asked with no model chosen: not pleased for a couple of seconds, glaring at the picker.
  const [grumpy, setGrumpy] = useState(false);
  const [glareAt, setGlareAt] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    if (!missingModelAt) return;
    const picker = document.querySelector("[data-model-picker]");
    const cat = catRef.current;
    if (picker && cat) setGlareAt(gazeToward(picker.getBoundingClientRect(), cat.getBoundingClientRect()));
    setGrumpy(true);
    const timer = setTimeout(() => setGrumpy(false), 2200);
    return () => clearTimeout(timer);
  }, [missingModelAt]);

  useLayoutEffect(() => {
    const textarea = document.querySelector<HTMLTextAreaElement>(COMPOSER_SELECTOR);
    const cat = catRef.current;
    if (!draft || !textarea || !cat) return;
    const caret = caretPoint(textarea);
    setLookAt(gazeToward(new DOMRect(caret.x, caret.y, 0, 0), cat.getBoundingClientRect()));
  }, [draft]);

  return (
    <div className="flex flex-col items-center animate-uzu-in">
      <span ref={catRef} className="mb-1 flex text-text-primary">
        <UzuCat
          state={grumpy ? "grumpy" : draft ? "watch" : "ready"}
          lookAt={grumpy ? glareAt : lookAt}
          anchorRef={catRef}
          size={112}
          label={grumpy ? "uzu, waiting for a model" : draft ? "uzu, watching you type" : "uzu, ready"}
        />
      </span>
      <div className="flex flex-col gap-1 text-center">
        <h2 className="text-[15px] font-[500] leading-[1.35] text-label-title text-balance">Start a private chat</h2>
        <p className="text-[13px] leading-[1.45] text-label-muted">It stays fully on your device</p>
      </div>
    </div>
  );
}
