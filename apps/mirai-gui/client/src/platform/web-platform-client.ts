import type { PlatformClient } from "./platform-client";
import type { ChatService } from "./services/chat";
import type { ModelsService } from "./services/models";
import type { SessionService } from "./services/session";
import { webSettings } from "./services/settings/web";
import { webDialogs } from "./services/dialogs/web";
import { webStorage } from "./services/storage/web";
import { webSystem } from "./services/system/web";
import { webSystemUi } from "./services/system-ui/web";
import { webUpdater } from "./services/updater/web";

// Browser build: the web app supplies inference, sessions and model downloads
// from uzu's WASM engine. Conversations stay on this device in IndexedDB and
// settings in localStorage.
export class WebPlatformClient implements PlatformClient {
  readonly storage = webStorage;
  readonly settings = webSettings;
  readonly system = webSystem;
  readonly systemUi = webSystemUi;
  readonly updater = webUpdater;
  readonly dialogs = webDialogs;

  constructor(
    readonly chat: ChatService,
    readonly session: SessionService,
    readonly models: ModelsService,
  ) {}
}
