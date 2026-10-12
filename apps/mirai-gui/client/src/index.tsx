import React from "react";

import ReactDOM from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";
import { createAppRouter } from "./router";
import "./index.css";
import { RuntimeLoader } from "@rive-app/react-canvas";
import riveWasmUrl from "@rive-app/canvas/rive.wasm?url";
import { initPlatform } from "./platform/platform-singleton";
import { TauriPlatformClient } from "./platform/tauri-platform-client";
import { WebPlatformClient } from "./platform/web-platform-client";
import { platformInfo } from "./platform/platform-info";
import { useAppStore } from "./stores/use-app-store";

RuntimeLoader.setWasmUrl(riveWasmUrl);

const appState = useAppStore.getState();
document.documentElement.classList.toggle("dark", appState.isDarkMode);

const getInitialPath = (): string => {
  if (!appState.skipWelcome) return "/welcome";
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
