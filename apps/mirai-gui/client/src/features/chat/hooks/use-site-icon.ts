import { useEffect, useState } from "react";
import { getPlatform } from "@/platform/platform-singleton";

const icons = new Map<string, Promise<string | null>>();

const iconOf = (host: string): Promise<string | null> => {
  let icon = icons.get(host);
  if (!icon) {
    icon = Promise.resolve()
      .then(() => getPlatform().chat.siteIcon(host))
      .catch(() => null)
      .then((url) => {
        // Kept only when found: a site that was unreachable is asked again next time.
        if (url === null) icons.delete(host);
        return url;
      });
    icons.set(host, icon);
  }
  return icon;
};

export function useSiteIcon(host: string): string | null {
  const [icon, setIcon] = useState<{ host: string; url: string | null } | null>(null);
  useEffect(() => {
    let current = true;
    void iconOf(host).then((url) => {
      if (current) setIcon({ host, url });
    });
    return () => {
      current = false;
    };
  }, [host]);
  return icon?.host === host ? icon.url : null;
}
