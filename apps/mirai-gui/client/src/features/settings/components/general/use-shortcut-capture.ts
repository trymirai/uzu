import { useEffect, useRef, useState } from "react";

const DISPLAY_KEYS: Record<string, string> = {
  Meta: "⌘",
  Control: "⌃",
  Alt: "⌥",
  Option: "⌥",
  Shift: "⇧",
  " ": "Space",
};

const CODE_KEYS: Record<string, string> = {
  Minus: "-",
  Equal: "=",
  BracketLeft: "[",
  BracketRight: "]",
  Semicolon: ";",
  Quote: "'",
  Backquote: "`",
  Backslash: "\\",
  Comma: ",",
  Period: ".",
  Slash: "/",
  Space: "Space",
};

function formatKeys(keys: string[]): string {
  return keys
    .map((k) => DISPLAY_KEYS[k] || k)
    .join(" ")
    .replace("  Space", " Space");
}

function acceleratorFrom(event: KeyboardEvent): string | null {
  const modifiers: string[] = [];
  if (event.metaKey) modifiers.push("Meta");
  if (event.ctrlKey) modifiers.push("Control");
  if (event.altKey) modifiers.push("Alt");
  if (event.shiftKey) modifiers.push("Shift");

  if (["Meta", "Control", "Alt", "Shift"].includes(event.key)) return null;

  const mappedCode = CODE_KEYS[event.code];
  let primary: string | null = null;
  if (event.code.startsWith("Key")) {
    primary = event.code.slice(3).toUpperCase();
  } else if (event.code.startsWith("Digit")) {
    primary = event.code.slice(5);
  } else if (mappedCode) {
    primary = mappedCode;
  } else if (/^F\d{1,2}$/.test(event.key)) {
    primary = event.key;
  } else if (event.key === " ") {
    primary = "Space";
  }
  if (!primary) return null;

  // Shift alone won't fire a global hotkey on macOS; keep capturing until
  // a system modifier is added rather than store a dead shortcut.
  if (!modifiers.some((m) => m === "Meta" || m === "Control" || m === "Alt")) return null;

  return [...modifiers]
    .map((p) => (p === "Meta" ? "CommandOrControl" : p))
    .concat(primary)
    .join("+");
}

export function useShortcutCapture(
  quickEntryShortcut: string | null,
  registerQuickEntryShortcut: (accelerator: string) => Promise<boolean>,
) {
  const [isCapturing, setIsCapturing] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!isCapturing) return;

    function onKeyDown(e: KeyboardEvent) {
      e.preventDefault();
      if (e.key === "Escape") {
        setIsCapturing(false);
        return;
      }
      const accelerator = acceleratorFrom(e);
      if (!accelerator) return;
      registerQuickEntryShortcut(accelerator).then(() => {
        setIsCapturing(false);
      });
    }

    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [isCapturing, registerQuickEntryShortcut]);

  useEffect(() => {
    if (!isCapturing) return;

    const handlePossibleOutside = (event: Event) => {
      const buttonEl = buttonRef.current;
      if (!buttonEl) {
        setIsCapturing(false);
        return;
      }
      if (!(event.target instanceof Node) || !buttonEl.contains(event.target)) {
        setIsCapturing(false);
      }
    };

    const handleWindowBlur = () => setIsCapturing(false);

    document.addEventListener("mousedown", handlePossibleOutside);
    document.addEventListener("touchstart", handlePossibleOutside);
    document.addEventListener("contextmenu", handlePossibleOutside);
    window.addEventListener("blur", handleWindowBlur);

    return () => {
      document.removeEventListener("mousedown", handlePossibleOutside);
      document.removeEventListener("touchstart", handlePossibleOutside);
      document.removeEventListener("contextmenu", handlePossibleOutside);
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, [isCapturing]);

  const shortcutDisplay = quickEntryShortcut
    ? formatKeys(quickEntryShortcut.split("+").map((p) => (p === "CommandOrControl" ? "Meta" : p)))
    : "";

  return { isCapturing, setIsCapturing, buttonRef, shortcutDisplay };
}
