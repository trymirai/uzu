import { isMacPlatform, isWindowsPlatform } from "@/utils/platform";

export const platformInfo = {
  isWeb: __PLATFORM__ === "web",
  isTauri: __PLATFORM__ === "tauri",
  features: {
    autoUpdate: __PLATFORM__ === "tauri" && isMacPlatform(),
    fileSystem: __PLATFORM__ === "tauri",
    logExport: __PLATFORM__ === "tauri",
    localModelDownloads: __PLATFORM__ === "tauri",
    autoEject: __PLATFORM__ === "tauri",
    cliInstall: __PLATFORM__ === "tauri" && isMacPlatform(),
    // Overlay title bar with in-content macOS traffic lights: UI must reserve
    // space on the top-left so controls don't sit under the traffic lights.
    nativeTitleBar: __PLATFORM__ === "tauri" && isMacPlatform(),
    // Frameless Windows window: the app draws the caption buttons on the
    // top-right, so UI must keep that corner free.
    windowControls: __PLATFORM__ === "tauri" && isWindowsPlatform(),
  },
} as const;
