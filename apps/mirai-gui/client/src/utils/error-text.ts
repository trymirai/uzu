export const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e));
