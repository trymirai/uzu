// Every write reads and rewrites the whole file, so two at once would lose each other's changes.
const chains = new Map<string, Promise<unknown>>();

export const withFileQueue = <T>(key: string, run: () => Promise<T>): Promise<T> => {
  const prev = chains.get(key) ?? Promise.resolve();
  const next = prev.then(run, run);
  const tail: Promise<unknown> = next
    .catch(() => undefined)
    .finally(() => {
      if (chains.get(key) === tail) chains.delete(key);
    });
  chains.set(key, tail);
  return next;
};
