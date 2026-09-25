import type { PlatformClient } from "./PlatformClient";
import { webChat } from "./services/chat/web";
import { webModels } from "./services/models/web";
import { webSession } from "./services/session/web";
import { webSettings } from "./services/settings/web";
import { webDialogs } from "./services/dialogs/web";
import { webStorage } from "./services/storage/web";
import { webSystem } from "./services/system/web";
import { webSystemUi } from "./services/system-ui/web";
import { webUpdater } from "./services/updater/web";

// Browser build: no engine and no backend. Inference is unavailable, settings
// live in localStorage, chats are not persisted at all.
export class WebPlatformClient implements PlatformClient {
  readonly chat = webChat;
  readonly session = webSession;
  readonly models = webModels;
  readonly storage = webStorage;
  readonly settings = webSettings;
  readonly system = webSystem;
  readonly systemUi = webSystemUi;
  readonly updater = webUpdater;
  readonly dialogs = webDialogs;
}
