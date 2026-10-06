import type { NavigationRequest } from "../../types/navigation";

export type SystemService = {
  onNavigationRequest(cb: (req: NavigationRequest) => void): () => void;
  /** App version, or null where the platform has no packaged version (web). */
  getAppVersion(): Promise<string | null>;
  /** Absolute path of the backend log file, or null where there is none. */
  getLogFilePath(): Promise<string | null>;
  /**
   * Install the mirai CLI wrapper now; also clears a persisted startup decline.
   */
  installCli(): Promise<"installed" | "already-installed" | "cancelled" | "unsupported">;
  /** "missing" is the only state where offering an install makes sense. */
  getCliStatus(): Promise<"missing" | "installed" | "foreign" | "unavailable">;
};
