import { useState } from "react";
import { useSiteIcon } from "../../hooks/use-site-icon";

const hostOf = (url: string): string => {
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
};

const hueOf = (site: string): number => [...site].reduce((hue, char) => (hue * 31 + char.charCodeAt(0)) % 360, 7);

export function SiteIcon({ url }: { url: string }) {
  const host = hostOf(url);
  const icon = useSiteIcon(host);
  const [broken, setBroken] = useState<string | null>(null);
  const site = host.replace(/^www\./, "");

  if (icon && broken !== icon) {
    return (
      <img
        src={icon}
        alt=""
        aria-hidden="true"
        onError={() => setBroken(icon)}
        className="size-3.5 shrink-0 rounded-[3px] bg-white object-contain"
      />
    );
  }
  return (
    <span
      aria-hidden="true"
      style={{ backgroundColor: `hsl(${hueOf(site)} 35% 42%)` }}
      className="grid size-3.5 shrink-0 place-items-center rounded-[3px] text-[8.5px] font-semibold uppercase leading-none text-white"
    >
      {site.charAt(0)}
    </span>
  );
}
