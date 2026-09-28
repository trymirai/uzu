import type { ChatService } from "./services/chat";
import type { DialogsService } from "./services/dialogs";
import type { ModelsService } from "./services/models";
import type { SessionService } from "./services/session";
import type { SettingsService } from "./services/settings";
import type { StorageService } from "./services/storage";
import type { SystemService } from "./services/system";
import type { SystemUiService } from "./services/system-ui";
import type { UpdaterService } from "./services/updater";

export type PlatformClient = {
  chat: ChatService;
  session: SessionService;
  models: ModelsService;
  storage: StorageService;
  settings: SettingsService;
  system: SystemService;
  systemUi: SystemUiService;
  updater: UpdaterService;
  dialogs: DialogsService;
};

export { navigationRequestTypes } from "./types/navigation";
export type { NavigationRequest } from "./types/navigation";
