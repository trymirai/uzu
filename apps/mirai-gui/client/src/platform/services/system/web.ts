import type { SystemService } from ".";
import { noopUnsubscribe } from "../shared/noop";

export const webSystem: SystemService = {
  onNavigationRequest: noopUnsubscribe,
  getAppVersion: () => Promise.resolve(null),
  getLogFilePath: () => Promise.resolve(null),
  installCli: () => Promise.resolve("unsupported" as const),
  getCliStatus: () => Promise.resolve("unavailable" as const),
};
