import { useEffect, useState } from "react";
import { platformInfo } from "@/platform/platform-info";

/**
 * True while the desktop app is in native full screen, where macOS hides the traffic lights. Asked
 * of the window itself on every resize: the window's size can't tell, since full screen may keep
 * the menu bar on screen and so cover less than the display.
 */
export function useNativeFullscreen(): boolean {
  const enabled = platformInfo.features.nativeTitleBar;
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let unlisten: (() => void) | undefined;
    let settle: ReturnType<typeof setTimeout> | undefined;

    void import("@tauri-apps/api/window").then(async ({ getCurrentWindow }) => {
      const appWindow = getCurrentWindow();
      const update = async () => {
        const next = await appWindow.isFullscreen();
        if (!cancelled) setFullscreen(next);
      };
      await update();
      // Resize events arrive while the full-screen animation is still running, so ask again once it
      // has settled.
      const stop = await appWindow.onResized(() => {
        void update();
        clearTimeout(settle);
        settle = setTimeout(() => void update(), 700);
      });
      if (cancelled) stop();
      else unlisten = stop;
    });

    return () => {
      cancelled = true;
      clearTimeout(settle);
      unlisten?.();
    };
  }, [enabled]);

  return fullscreen;
}
