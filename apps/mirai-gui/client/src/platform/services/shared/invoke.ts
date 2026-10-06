import { invoke as tauriInvoke, type InvokeArgs } from "@tauri-apps/api/core";

// The backend serializes its errors as plain strings, so a rejected invoke
// carries a string, while the client reads failures through Error.
export const invoke = <T>(command: string, args?: InvokeArgs): Promise<T> =>
  tauriInvoke<T>(command, args).catch((error: unknown) => {
    throw error instanceof Error ? error : new Error(String(error));
  });
