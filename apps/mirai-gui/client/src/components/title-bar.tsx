import { Copy, Minus, Square, X } from "lucide-react";
import { useEffect, useState } from "react";
import { getPlatform } from "@/platform/platform-singleton";
import { isWindowsPlatform } from "@/utils/platform";

// The menus' Ctrl shortcuts, by item id in desktop/src/system_ui.rs. WebView2 keeps keys from the window, so the
// menus never see them; the page runs them instead. GTK runs them itself.
const MENU_SHORTCUTS: Record<string, string> = {
  n: "menu:new-chat",
  ",": "menu:settings",
  "=": "menu:zoom-in",
  "-": "menu:zoom-out",
  "0": "menu:zoom-reset",
};

/**
 * The frameless window's title bar on Windows and Linux, like VS Code's: the app icon, the app menus
 * (each opens the native menu under its title), a drag area, and the caption buttons, dimmed while
 * the window is inactive. Windows draws them with its Segoe Fluent Icons glyphs (Segoe MDL2 Assets on
 * Windows 10), which Linux lacks. The sidebar puts its button in `start`, as the Mac app has it
 * beside the traffic lights.
 */
export function TitleBar({ start }: { start: (element: HTMLElement | null) => void }) {
  const systemUi = getPlatform().systemUi;
  const isWindows = isWindowsPlatform();
  const [titles, setTitles] = useState<string[]>([]);
  const [maximized, setMaximized] = useState(false);
  const [active, setActive] = useState(() => document.hasFocus());
  useEffect(() => void systemUi.menuTitles().then(setTitles), [systemUi]);
  useEffect(() => systemUi.onWindowMaximizedChange(setMaximized), [systemUi]);
  useEffect(() => {
    if (!isWindows) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const id = event.ctrlKey && !event.altKey && !event.shiftKey ? MENU_SHORTCUTS[event.key] : undefined;
      if (!id) return;
      event.preventDefault();
      void systemUi.runMenuItem(id);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isWindows, systemUi]);
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
      <div ref={start} className="flex items-center" />
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
          { label: "Minimize", glyph: "\uE921", Icon: Minus, hover: "hover:bg-bg-hover", run: systemUi.minimizeWindow },
          {
            label: maximized ? "Restore" : "Maximize",
            glyph: maximized ? "\uE923" : "\uE922",
            Icon: maximized ? Copy : Square,
            hover: "hover:bg-bg-hover",
            run: systemUi.toggleMaximizeWindow,
          },
          {
            label: "Close",
            glyph: "\uE8BB",
            Icon: X,
            hover: "hover:bg-[#c42b1c] hover:text-white",
            run: systemUi.closeWindow,
          },
        ].map((button) => (
          <button
            key={button.label}
            type="button"
            aria-label={button.label}
            className={`flex h-full w-[46px] items-center justify-center text-[10px] transition-colors duration-100 ${textColor} ${button.hover}`}
            onClick={() => void button.run()}
          >
            {isWindows ? button.glyph : <button.Icon className="size-3.5" strokeWidth={1.5} />}
          </button>
        ))}
      </div>
    </div>
  );
}
