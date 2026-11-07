export const platformInfo = {
  isWeb: __PLATFORM__ === "web",
  isTauri: __PLATFORM__ === "tauri",
  /** The browser build with a pretend engine, for design work and recordings. */
  isDemo: __PLATFORM__ === "demo",
  features: {
    autoUpdate: __PLATFORM__ === "tauri",
    fileSystem: __PLATFORM__ === "tauri",
    logExport: __PLATFORM__ === "tauri",
    localModelDownloads: __PLATFORM__ === "tauri" || __PLATFORM__ === "demo",
    autoEject: __PLATFORM__ === "tauri",
    cliInstall: __PLATFORM__ === "tauri",
    // Overlay title bar with in-content macOS traffic lights: UI must reserve
    // space on the top-left so controls don't sit under the traffic lights.
    nativeTitleBar: __PLATFORM__ === "tauri",
  },
} as const;
