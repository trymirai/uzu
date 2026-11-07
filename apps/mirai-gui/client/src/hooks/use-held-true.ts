import { useEffect, useState } from "react";

/**
 * `value`, except that once true it stays true until it has been false for `holdMs`. For states
 * that hand off from one task to the next (the answer streams, then the chat is named): without the
 * hold the gap between them shows as a one-frame flicker of whatever the state controls.
 */
export function useHeldTrue(value: boolean, holdMs: number): boolean {
  const [held, setHeld] = useState(value);
  useEffect(() => {
    if (value) {
      setHeld(true);
      return;
    }
    const timer = setTimeout(() => setHeld(false), holdMs);
    return () => clearTimeout(timer);
  }, [value, holdMs]);
  return value || held;
}
