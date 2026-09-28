import { useSyncExternalStore } from "react";

let shiftHeld = false;
const listeners = new Set<() => void>();

const setShiftHeld = (value: boolean) => {
  if (shiftHeld === value) return;
  shiftHeld = value;
  listeners.forEach((notify) => notify());
};

if (typeof window !== "undefined") {
  window.addEventListener("keydown", (e) => setShiftHeld(e.shiftKey), true);
  // e.shiftKey stays true while the sibling Shift of a double-hold is down.
  window.addEventListener("keyup", (e) => setShiftHeld(e.shiftKey && e.key !== "Shift"), true);
  // Keyup never arrives if focus left the window while Shift was down.
  window.addEventListener("blur", () => setShiftHeld(false));
}

const subscribe = (notify: () => void): (() => void) => {
  listeners.add(notify);
  return () => {
    listeners.delete(notify);
  };
};

const getSnapshot = (): boolean => shiftHeld;
const getServerSnapshot = (): boolean => false;

/** True while Shift is held; window-level singleton, safe for many subscribers. */
export const useShiftHeld = (): boolean => useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
