import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "../shared/invoke";
import type { SystemUiService } from ".";

export const tauriSystemUi: SystemUiService = {
  setWindowTheme: (theme) => invoke<boolean>("set_window_theme", { theme }),
  getZoom: () => invoke<number>("zoom_get").catch(() => 1),
  stepZoom: (step) => invoke<number>("zoom_step", { step }),
  onZoomChange: (cb) => {
    let disposed = false;
    let off: (() => void) | undefined;
    // listen() resolves after the caller may already have unsubscribed.
    void listen<number>("app:zoom", (event) => cb(event.payload))
      .then((stop) => (disposed ? stop() : (off = stop)))
      .catch(() => undefined);
    return () => {
      disposed = true;
      off?.();
    };
  },
  menuTitles: () => invoke<string[]>("menu_titles"),
  openMenu: (index, left, bottom) => invoke("open_menu", { index, left, bottom }),
  runMenuItem: (id) => invoke("run_menu_item", { id }),
  minimizeWindow: () => getCurrentWindow().minimize(),
  toggleMaximizeWindow: () => getCurrentWindow().toggleMaximize(),
  closeWindow: () => getCurrentWindow().close(),
  onWindowMaximizedChange: (cb) => {
    const appWindow = getCurrentWindow();
    let disposed = false;
    let off: (() => void) | undefined;
    const update = () => void appWindow.isMaximized().then((maximized) => disposed || cb(maximized));
    update();
    // onResized() resolves after the caller may already have unsubscribed.
    void appWindow
      .onResized(update)
      .then((stop) => (disposed ? stop() : (off = stop)))
      .catch(() => undefined);
    return () => {
      disposed = true;
      off?.();
    };
  },
};
