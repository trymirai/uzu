import { ArrowUpRight, ChevronRight } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { platformInfo } from "@/platform/platform-info";
import { UzuCat } from "@/components/uzu-cat";
import { SettingGroup } from "./general/setting-row";
import { Button } from "@/components/ui/button";
const connectLinks = [
  { title: "Website", url: "https://trymirai.com/" },
  { title: "About us", url: "https://trymirai.com/about-us" },
  { title: "GitHub", url: "https://github.com/trymirai" },
  { title: "Discord", url: "https://discord.gg/trymirai" },
];

const moreLinks = [
  { title: "Metrics", description: "uzu against MLX, MTPLX and llama.cpp.", url: "https://trymirai.com/metrics" },
  {
    title: "Models",
    description: "25 Mirai checkpoints with per-device benchmarks.",
    url: "https://trymirai.com/models",
  },
  {
    title: "Command-line interface",
    description: "Send messages to a model interactively, or start a local server.",
    url: "https://github.com/trymirai/uzu#cli",
  },
  {
    title: "Rust inference engine",
    description: "Runs models on the hardware they were built for.",
    url: "https://github.com/trymirai/uzu",
  },
  { title: "Careers", description: "Explore open positions.", url: "https://trymirai.com/careers" },
];

const legalLinks = [
  {
    title: "Terms of Service",
    url: "https://artifacts.trymirai.com/legal/Mirai_Tech_Terms_of_Use.pdf",
  },
  {
    title: "Privacy Policy",
    url: "https://artifacts.trymirai.com/legal/Mirai_Tech_Privacy_Policy.pdf",
  },
];

function LinkRow({ title, description, url }: { title: string; description?: string; url: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-center justify-between gap-4 px-4 py-3.5 transition-colors duration-150 ease-out hover:bg-surface-tertiary outline-hidden focus-visible:shadow-focus"
    >
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="text-[14px] font-[500] leading-[1.35] text-label-title">{title}</span>
        {description ? (
          <span className="text-[13px] leading-[1.45] text-label-muted text-pretty">{description}</span>
        ) : null}
      </span>
      <ArrowUpRight
        aria-hidden
        className="size-4 shrink-0 text-label-muted transition-[color,translate] duration-150 ease-out group-hover:text-label-title group-hover:translate-x-px group-hover:-translate-y-px"
      />
    </a>
  );
}

/** Opens the first-run welcome again, to see what a new install shows. */
function WelcomeRow() {
  return (
    <Link
      to="/welcome"
      className="group flex items-center justify-between gap-4 px-4 py-3.5 transition-colors duration-150 ease-out hover:bg-surface-tertiary outline-hidden focus-visible:shadow-focus"
    >
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="text-[14px] font-[500] leading-[1.35] text-label-title">Welcome screen</span>
        <span className="text-[13px] leading-[1.45] text-label-muted text-pretty">What a new install shows first.</span>
      </span>
      <ChevronRight
        aria-hidden
        className="size-4 shrink-0 text-label-muted transition-[color,translate] duration-150 ease-out group-hover:text-label-title group-hover:translate-x-px"
      />
    </Link>
  );
}

export default function AboutTab() {
  return (
    <div className="flex flex-col gap-7 pb-8">
      <div className="flex flex-col items-center text-center">
        <div className="mb-1">
          <UzuCat state="welcome" size={96} className="text-text-primary" label="uzu, saying hello" />
        </div>
        <h3 className="max-w-[440px] text-[15px] font-[500] leading-[1.35] text-label-title text-balance">
          Built by a small, senior team building the full on-device stack to achieve realtime local intelligence.
        </h3>
        <Button
          size="sm"
          href="https://x.com/trymirai"
          target="_blank"
          className="mt-5 rounded-full bg-blue/10 text-blue hover:bg-blue/20 active:bg-blue/25"
        >
          Connect with us on X
        </Button>
      </div>

      <SettingGroup title="Connect">
        {connectLinks.map((link) => (
          <LinkRow key={link.url} title={link.title} url={link.url} />
        ))}
      </SettingGroup>

      <SettingGroup title="More from Mirai Labs">
        {moreLinks.map((link) => (
          <LinkRow key={link.url} title={link.title} description={link.description} url={link.url} />
        ))}
      </SettingGroup>

      <SettingGroup title="Legal">
        {legalLinks.map((link) => (
          <LinkRow key={link.url} title={link.title} url={link.url} />
        ))}
      </SettingGroup>

      {/* The welcome belongs to the desktop app, which is where it appears on first launch. */}
      {platformInfo.features.localModelDownloads ? (
        <SettingGroup title="Preview">
          <WelcomeRow />
        </SettingGroup>
      ) : null}
    </div>
  );
}
