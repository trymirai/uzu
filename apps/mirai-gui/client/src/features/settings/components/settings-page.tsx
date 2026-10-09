import { Link, useSearch } from "@tanstack/react-router";
import GeneralTab from "./general-tab";
import AboutTab from "./about-tab";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { twMerge } from "tailwind-merge";

export type SettingsTab = "general" | "about";

export function SettingsPage() {
  const { tab } = useSearch({ from: "/settings" });
  const sections: { key: SettingsTab; label: string }[] = [
    { key: "general", label: "General" },
    { key: "about", label: "About" },
  ];

  const tabComponents: Record<SettingsTab, React.ReactNode> = {
    general: <GeneralTab />,
    about: <AboutTab />,
  };

  return (
    <div className="h-full min-h-0 w-full bg-background text-label-title pt-4">
      <div className="h-full min-h-0 flex flex-col">
        <div className="px-5 shrink-0 z-10 bg-background">
          <PageHeader title={<h1 className="text-[15px] leading-[150%] font-medium mb-4">Settings</h1>} />
        </div>
        <div className="h-[1px] shrink-0 bg-cell-border" />

        <div className="lg:hidden px-5 py-4 shrink-0 z-10 bg-background">
          <div className="flex gap-2 overflow-x-auto overscroll-x-contain overscroll-y-auto thin-scrollbar">
            {sections.map((s) => (
              <Link key={s.key} to="/settings" search={{ tab: s.key }} className="flex-shrink-0">
                <Button kind={s.key === tab ? "primary" : "ghost"} size="sm" className="whitespace-nowrap">
                  {s.label}
                </Button>
              </Link>
            ))}
          </div>
        </div>

        <div className="flex flex-1 min-h-0">
          <aside className="hidden lg:block w-[152px] flex-shrink-0 h-full pt-2 overflow-y-auto thin-scrollbar border-r border-cell-border">
            <nav className="flex flex-col gap-2">
              {sections.map((s) => (
                <Link key={s.key} to="/settings" search={{ tab: s.key }} className="flex px-3">
                  <div
                    className={`flex items-center gap-3 px-2 py-[6px] w-full rounded-md ${s.key === tab ? "bg-bg-hover dark:bg-bg-hover" : ""} hover:bg-bg-hover `}
                  >
                    <span className="text-[13px] font-[350] leading-[150%] text-label-title">{s.label}</span>
                  </div>
                </Link>
              ))}
            </nav>
          </aside>

          <section
            className={twMerge(
              "flex-1 min-w-0 lg:px-0 h-full overflow-y-auto overscroll-y-contain thin-scrollbar [scrollbar-gutter:stable]",
              "pt-5 lg:pt-5",
            )}
          >
            {tabComponents[tab]}
          </section>
        </div>
      </div>
    </div>
  );
}
