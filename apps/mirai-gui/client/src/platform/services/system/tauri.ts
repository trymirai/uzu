import { getVersion } from "@tauri-apps/api/app";
import { invoke } from "../shared/invoke";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { navigationRequestTypes, type NavigationRequest } from "../../types/navigation";
import { isMacPlatform } from "@/utils/platform";
import type { SystemService } from ".";

// The context menu outside text fields (WebView2's, and WebKitGTK's Back, Forward, Stop and Reload) and the reload, find,
// print, save, history and caret-browsing keys make no sense in the app; development builds keep them. AltGr reaches
// WebView2 as Ctrl+Alt and types text.
const BROWSER_KEYS = new Set(["F3", "F5", "F7", "BrowserBack", "BrowserForward", "BrowserRefresh"]);
const BROWSER_CTRL_LETTERS = new Set(["f", "g", "p", "r", "s", "u"]);
export const disableBrowserBehaviour = (): void => {
  if (!import.meta.env.PROD || isMacPlatform()) return;
  window.addEventListener(
    "contextmenu",
    (event) => {
      if (!(event.target instanceof Element && event.target.closest("input, textarea, [contenteditable]"))) {
        event.preventDefault();
      }
    },
    true,
  );
  window.addEventListener(
    "keydown",
    (event) => {
      // Non-Latin layouts report the localized character; fall back to the physical key.
      const letter = /^[a-z]$/i.test(event.key)
        ? event.key.toLowerCase()
        : event.code.replace(/^Key/, "").toLowerCase();
      const history = event.altKey && (event.key === "ArrowLeft" || event.key === "ArrowRight");
      const browserShortcut = event.ctrlKey && !event.altKey && BROWSER_CTRL_LETTERS.has(letter);
      if (BROWSER_KEYS.has(event.key) || history || browserShortcut) {
        event.preventDefault();
      }
    },
    true,
  );
};

// WKWebView has no window-open handler wired up, so target="_blank" anchors
// would silently do nothing; intercept clicks and open externally via Rust.
let interceptingLinks = false;
export const ensureExternalLinkInterception = (): void => {
  if (interceptingLinks) return;
  interceptingLinks = true;
  document.addEventListener(
    "click",
    (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const anchor = target.closest("a[href]");
      if (!(anchor instanceof HTMLAnchorElement)) return;
      const href = anchor.href;
      if (!/^https?:\/\//.test(href)) return;
      if (new URL(href).origin === window.location.origin) return;
      event.preventDefault();
      void invoke("open_external", { url: href }).catch(() => undefined);
    },
    true,
  );
};

export const tauriSystem: SystemService = {
  onNavigationRequest: (cb) => {
    const offs: Array<() => void> = [];
    let disposed = false;
    // listen() resolves after the caller may already have unsubscribed.
    const track = (p: Promise<() => void>): void => {
      void p.then((off) => (disposed ? off() : offs.push(off))).catch(() => undefined);
    };
    track(
      listen("app:new-chat", () => {
        cb({ type: navigationRequestTypes.newChat } satisfies NavigationRequest);
      }),
    );
    track(
      listen("app:open-preferences", () => {
        cb({ type: navigationRequestTypes.openPreferences } satisfies NavigationRequest);
      }),
    );
    // The macOS app menu owns these shortcuts; Windows and Linux have no menu bar.
    if (!isMacPlatform()) {
      const onKeyDown = (event: KeyboardEvent): void => {
        if (!event.ctrlKey || event.shiftKey || event.altKey || event.metaKey) return;
        // Non-Latin layouts report the localized character; fall back to the physical key.
        const key = /^[\x20-\x7e]$/.test(event.key) ? event.key.toLowerCase() : event.code;
        if (key === "n" || key === "KeyN") cb({ type: navigationRequestTypes.newChat } satisfies NavigationRequest);
        else if (key === "," || key === "Comma")
          cb({ type: navigationRequestTypes.openPreferences } satisfies NavigationRequest);
        else return;
        event.preventDefault();
      };
      window.addEventListener("keydown", onKeyDown);
      offs.push(() => window.removeEventListener("keydown", onKeyDown));
    }
    return () => {
      disposed = true;
      offs.forEach((off) => off());
    };
  },
  getAppVersion: () => getVersion().catch(() => null),
  getLogFilePath: () => invoke<string>("get_log_file_path").catch(() => null),
  installCli: () => invoke<"installed" | "already-installed" | "cancelled">("cli_install"),
  getCliStatus: () =>
    invoke<"missing" | "installed" | "foreign" | "unavailable">("cli_status").catch(() => "unavailable" as const),
  minimizeWindow: () => getCurrentWindow().minimize(),
  toggleMaximizeWindow: () => getCurrentWindow().toggleMaximize(),
  closeWindow: () => getCurrentWindow().close(),
  onWindowMaximizedChange: (cb) => {
    const window = getCurrentWindow();
    let disposed = false;
    const report = (): void => {
      void window
        .isMaximized()
        .then((maximized) => {
          if (!disposed) cb(maximized);
        })
        .catch(() => undefined);
    };
    report();
    const unlisten = window.onResized(report);
    return () => {
      disposed = true;
      void unlisten.then((off) => off()).catch(() => undefined);
    };
  },
  onCloseRequested: (cb) => {
    // On macOS closing hides the window and the backend owns that.
    if (isMacPlatform()) return () => {};
    // A rejected handler would keep the window open.
    const unlisten = getCurrentWindow().onCloseRequested(() =>
      cb().catch((error: unknown) => console.error("[window] close handler failed", error)),
    );
    return () => void unlisten.then((off) => off()).catch(() => undefined);
  },
};
