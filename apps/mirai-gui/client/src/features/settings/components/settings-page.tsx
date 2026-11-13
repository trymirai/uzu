import { useNavigate, useSearch } from "@tanstack/react-router";
import type { CSSProperties } from "react";
import GeneralTab from "./general-tab";
import AboutTab from "./about-tab";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { PageScroller, PageTitleBar } from "@/components/page-title-bar";
import { useAppStore } from "@/stores/use-app-store";

export type SettingsTab = "general" | "about";

const SECTIONS: { value: SettingsTab; label: string }[] = [
  { value: "general", label: "General" },
  { value: "about", label: "About" },
];

/** Set like the chats page: the title at the top of the column, the two sections as one switch. */
export function SettingsPage() {
  const { tab } = useSearch({ from: "/settings" });
  const navigate = useNavigate();
  const chatWidth = useAppStore((s) => s.chatWidth);

  return (
    <div
      className="h-full min-h-0 w-full bg-background text-label-title"
      style={{ "--chat-width": `${chatWidth}px` } as CSSProperties}
    >
      <PageScroller
        header={
          <PageTitleBar className="justify-between">
            <h1 className="text-[20px] font-[600] leading-[1.2] tracking-[-0.01em] text-label-title">Settings</h1>
            <SegmentedControl
              ariaLabel="Settings section"
              value={tab}
              onChange={(value) => navigate({ to: "/settings", search: { tab: value as SettingsTab } })}
              options={SECTIONS}
            />
          </PageTitleBar>
        }
      >
        <section key={tab} className="px-5 pt-4 w-full max-w-[calc(var(--chat-width)+2.5rem)] mx-auto animate-uzu-in">
          {tab === "about" ? <AboutTab /> : <GeneralTab />}
        </section>
      </PageScroller>
    </div>
  );
}
