import React from "react";

import ReactDOM from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";
import "@/ui-kit/index.css";
import { createAppRouter } from "./router";
import "./index.css";
import { RuntimeLoader } from "@rive-app/react-canvas";
import riveWasmUrl from "@rive-app/canvas/rive.wasm?url";
import { initPlatform } from "./platform/platformSingleton";
import { TauriPlatformClient } from "./platform/TauriPlatformClient";
import { WebPlatformClient } from "./platform/WebPlatformClient";
import { platformInfo } from "./platform/platformInfo";
import { APP_STORE_KEY } from "./stores/migrateAppStorage";

RuntimeLoader.setWasmUrl(riveWasmUrl);

type PersistedAppState = { isDarkMode?: unknown; skipWelcome?: unknown };

const readPersistedAppState = (): PersistedAppState => {
  try {
    const stored = localStorage.getItem(APP_STORE_KEY);
    return stored ? (JSON.parse(stored)?.state ?? {}) : {};
  } catch {
    return {};
  }
};

const persistedAppState = readPersistedAppState();

document.documentElement.classList.toggle("dark", persistedAppState.isDarkMode !== false);

const getInitialPath = (): string => {
  if (!persistedAppState.skipWelcome) return "/welcome";
  return platformInfo.isTauri ? "/local-models" : "/chats";
};

if (platformInfo.isTauri) {
  initPlatform(new TauriPlatformClient());
} else {
  initPlatform(new WebPlatformClient());
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={createAppRouter(getInitialPath())} />
  </React.StrictMode>,
);
