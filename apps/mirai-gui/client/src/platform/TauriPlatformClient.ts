import type { PlatformClient } from "./PlatformClient";
import { tauriChat } from "./services/chat/tauri";
import { tauriDialogs } from "./services/dialogs/tauri";
import { tauriModels } from "./services/models/tauri";
import { tauriSession } from "./services/session/tauri";
import { tauriSettings } from "./services/settings/tauri";
import { tauriStorage } from "./services/storage/tauri";
import { ensureExternalLinkInterception, tauriSystem } from "./services/system/tauri";
import { tauriSystemUi } from "./services/system-ui/tauri";
import { tauriUpdater } from "./services/updater/tauri";

export class TauriPlatformClient implements PlatformClient {
  readonly chat = tauriChat;
  readonly session = tauriSession;
  readonly models = tauriModels;
  readonly storage = tauriStorage;
  readonly settings = tauriSettings;
  readonly system = tauriSystem;
  readonly systemUi = tauriSystemUi;
  readonly updater = tauriUpdater;
  readonly dialogs = tauriDialogs;

  constructor() {
    ensureExternalLinkInterception();
  }
}
