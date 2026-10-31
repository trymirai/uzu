import { useEffect, useState } from "react";
import { platformInfo } from "@/platform/platform-info";
import { getPlatform } from "@/platform/platform-singleton";

/** Width of the three caption buttons, if drawn; headers keep their content out of it. */
export const HEADER_RIGHT_RESERVE_PX = platformInfo.features.windowControls ? 3 * 46 : 0;

const BUTTON =
  "overlay-button flex h-8 w-[46px] items-center justify-center text-[10px] transition-colors duration-100";

// Windows 11 caption buttons, drawn with the system's Segoe Fluent Icons glyphs (Segoe MDL2 Assets on Windows 10).
// Like the system's, their glyphs dim while the window is inactive.
export const WindowControls = () => {
  const [maximized, setMaximized] = useState(false);
  const [active, setActive] = useState(() => document.hasFocus());
  useEffect(() => getPlatform().system.onWindowMaximizedChange(setMaximized), []);
  useEffect(() => {
    const update = () => setActive(document.hasFocus());
    window.addEventListener("focus", update);
    window.addEventListener("blur", update);
    return () => {
      window.removeEventListener("focus", update);
      window.removeEventListener("blur", update);
    };
  }, []);
  const system = getPlatform().system;
  const glyph = active ? "text-label-title" : "text-label-title/40";

  return (
    <div
      className="fixed top-0 right-0 z-[60] flex"
      style={{ fontFamily: '"Segoe Fluent Icons", "Segoe MDL2 Assets"' }}
    >
      <button
        type="button"
        aria-label="Minimize"
        className={`${BUTTON} ${glyph} hover:bg-bg-hover hover:text-label-title`}
        onClick={() => void system.minimizeWindow()}
      >
        {"\uE921"}
      </button>
      <button
        type="button"
        aria-label={maximized ? "Restore" : "Maximize"}
        className={`${BUTTON} ${glyph} hover:bg-bg-hover hover:text-label-title`}
        onClick={() => void system.toggleMaximizeWindow()}
      >
        {maximized ? "\uE923" : "\uE922"}
      </button>
      <button
        type="button"
        aria-label="Close"
        className={`${BUTTON} ${glyph} hover:bg-[#c42b1c] hover:text-white`}
        onClick={() => void system.closeWindow()}
      >
        {"\uE8BB"}
      </button>
    </div>
  );
};
