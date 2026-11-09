import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";
import { RuntimeLoader } from "@rive-app/react-canvas";
import riveWasmUrl from "@rive-app/canvas/rive.wasm?url";
import type { AppRouter } from "./router";
import type { PlatformClient } from "./platform/platform-client";
import { initPlatform } from "./platform/platform-singleton";
import { useAppStore } from "./stores/use-app-store";

export function mountApp(platform: PlatformClient, createRouter: () => AppRouter) {
  const root = document.getElementById("root");
  if (!root) throw new Error("Root element is missing");

  RuntimeLoader.setWasmUrl(riveWasmUrl);
  initPlatform(platform);
  document.documentElement.classList.toggle("dark", useAppStore.getState().isDarkMode);
  const router = createRouter();

  createRoot(root).render(
    <StrictMode>
      <RouterProvider router={router} />
    </StrictMode>,
  );
}
