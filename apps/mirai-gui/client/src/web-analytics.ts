import type { AppRouter } from "./router";

const measurementId = "G-QNMKW6JT1G";
const siteOrigin = "https://uzu.run";

type GoogleTag = (command: string, ...parameters: unknown[]) => void;
type AnalyticsWindow = Window & { dataLayer?: unknown[]; gtag?: GoogleTag };

const pageTitles: Record<string, string> = {
  "/": "Home",
  "/welcome": "Welcome",
  "/local-models": "Models",
  "/chats": "Chats",
  "/chat": "Chat",
  "/settings": "Settings",
  "/not-found": "Not found",
};

function pageLocation(pathname: string): string {
  // Only send known screens, never chat IDs or arbitrary URL paths.
  const path = pathname.startsWith("/chat/") ? "/chat" : pathname;
  return siteOrigin + (Object.prototype.hasOwnProperty.call(pageTitles, path) ? path : "/not-found");
}

function initialReferrer(): string {
  if (!document.referrer) return "";
  try {
    const url = new URL(document.referrer);
    return url.origin === siteOrigin ? pageLocation(url.pathname) : url.origin + "/";
  } catch {
    return "";
  }
}

export function initializeWebAnalytics(router: AppRouter): void {
  if (__PLATFORM__ !== "web" || !import.meta.env.PROD || window.location.origin !== siteOrigin) {
    return;
  }

  const analyticsWindow = window as AnalyticsWindow;
  const dataLayer = (analyticsWindow.dataLayer ??= []);
  // gtag.js consumes the arguments object used by Google's standard snippet.
  const gtag: GoogleTag = function () {
    // eslint-disable-next-line prefer-rest-params
    dataLayer.push(arguments);
  };
  analyticsWindow.gtag = gtag;

  let previousPath: string | undefined;
  let referrer = initialReferrer();
  const pageParameters = (pathname: string) => {
    const location = pageLocation(pathname);
    return {
      page_location: location,
      page_title: `uzu.run - ${pageTitles[location.slice(siteOrigin.length)]}`,
      page_referrer: referrer,
    };
  };

  gtag("js", new Date());
  gtag("config", measurementId, {
    ...pageParameters(router.state.location.pathname),
    send_page_view: false,
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  });

  // Enhanced measurement must be disabled for this stream to avoid duplicate
  // history events and automatic form/search collection.
  router.subscribe("onResolved", ({ toLocation }) => {
    if (toLocation.pathname === previousPath) return;
    const parameters = pageParameters(toLocation.pathname);
    gtag("config", measurementId, { ...parameters, update: true, send_page_view: false });
    gtag("event", "page_view", { ...parameters, send_to: measurementId });
    previousPath = toLocation.pathname;
    referrer = parameters.page_location;
  });

  const script = document.createElement("script");
  script.async = true;
  script.crossOrigin = "anonymous";
  script.referrerPolicy = "origin";
  script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
  document.head.appendChild(script);
}
