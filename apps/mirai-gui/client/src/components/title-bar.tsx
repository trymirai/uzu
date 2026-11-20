import { useEffect, useState } from "react";
import { getPlatform } from "@/platform/platform-singleton";

// The menus' Ctrl shortcuts, by item id in desktop/src/system_ui.rs. WebView2 keeps keys from the window, so the
// menus never see them; the page runs them instead.
const MENU_SHORTCUTS: Record<string, string> = {
  n: "menu:new-chat",
  ",": "menu:settings",
  "=": "menu:zoom-in",
  "-": "menu:zoom-out",
  "0": "menu:zoom-reset",
};

/**
 * The frameless Windows window's title bar, like VS Code's: the app icon, the app menus (each opens
 * the native menu under its title), a drag area, and the caption buttons, drawn with the system's
 * Segoe Fluent Icons glyphs (Segoe MDL2 Assets on Windows 10) and dimmed while the window is inactive.
 */
export function TitleBar() {
  const systemUi = getPlatform().systemUi;
  const [titles, setTitles] = useState<string[]>([]);
  const [maximized, setMaximized] = useState(false);
  const [active, setActive] = useState(() => document.hasFocus());
  useEffect(() => void systemUi.menuTitles().then(setTitles), [systemUi]);
  useEffect(() => systemUi.onWindowMaximizedChange(setMaximized), [systemUi]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const id = event.ctrlKey && !event.altKey && !event.shiftKey ? MENU_SHORTCUTS[event.key] : undefined;
      if (!id) return;
      event.preventDefault();
      void systemUi.runMenuItem(id);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [systemUi]);
  useEffect(() => {
    const update = () => setActive(document.hasFocus());
    window.addEventListener("focus", update);
    window.addEventListener("blur", update);
    return () => {
      window.removeEventListener("focus", update);
      window.removeEventListener("blur", update);
    };
  }, []);
  const textColor = active ? "text-label-title" : "text-label-title/40";

  return (
    <div data-tauri-drag-region className="fixed inset-x-0 top-0 z-[60] flex h-8 select-none">
      <img src="/icons/icon_mirai_app.png" alt="" draggable={false} className="mx-2.5 size-4 self-center" />
      {titles.map((title, index) => (
        <button
          key={title}
          type="button"
          className={`my-1 rounded-md px-2 text-[13px] ${textColor} hover:bg-bg-hover`}
          onClick={(event) => {
            const { left, bottom } = event.currentTarget.getBoundingClientRect();
            void systemUi.openMenu(index, left, bottom);
          }}
        >
          {title}
        </button>
      ))}
      <div data-tauri-drag-region className="flex-1" />
      <div className="flex" style={{ fontFamily: '"Segoe Fluent Icons", "Segoe MDL2 Assets"' }}>
        {[
          { label: "Minimize", glyph: "\uE921", hover: "hover:bg-bg-hover", run: systemUi.minimizeWindow },
          {
            label: maximized ? "Restore" : "Maximize",
            glyph: maximized ? "\uE923" : "\uE922",
            hover: "hover:bg-bg-hover",
            run: systemUi.toggleMaximizeWindow,
          },
          { label: "Close", glyph: "\uE8BB", hover: "hover:bg-[#c42b1c] hover:text-white", run: systemUi.closeWindow },
        ].map((button) => (
          <button
            key={button.label}
            type="button"
            aria-label={button.label}
            className={`flex h-full w-[46px] items-center justify-center text-[10px] transition-colors duration-100 ${textColor} ${button.hover}`}
            onClick={() => void button.run()}
          >
            {button.glyph}
          </button>
        ))}
      </div>
    </div>
  );
}
