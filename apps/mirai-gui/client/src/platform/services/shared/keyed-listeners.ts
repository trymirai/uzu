export const createKeyedListeners = () => {
  const byKey = new Map<string, Set<(payload: never) => void>>();

  const on = (key: string, cb: (payload: never) => void): (() => void) => {
    const set = byKey.get(key) ?? new Set();
    set.add(cb);
    byKey.set(key, set);
    return () => {
      set.delete(cb);
      if (set.size === 0) byKey.delete(key);
    };
  };

  const emit = (key: string, payload: unknown): void => {
    byKey.get(key)?.forEach((cb) => {
      try {
        (cb as (p: unknown) => void)(payload);
      } catch (error) {
        // The other listeners still get the event.
        console.error("[platform] listener failed", { key }, error);
      }
    });
  };

  return { on, emit };
};
