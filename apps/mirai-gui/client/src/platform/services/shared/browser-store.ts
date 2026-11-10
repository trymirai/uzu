export const readJson = <T>(key: string, fallback: T): T => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
};

export const writeJson = (key: string, value: unknown): void => {
  // Callers must observe a rejected save so their UI can restore the persisted
  // preference instead of treating blocked or full storage as a successful write.
  localStorage.setItem(key, JSON.stringify(value));
};
