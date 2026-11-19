import { listen } from "@tauri-apps/api/event";
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
};
