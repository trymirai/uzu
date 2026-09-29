import { Link, useSearch } from "@tanstack/react-router";
import GeneralTab from "./general-tab";
import PrivacyTab from "./privacy-tab";
import AboutTab from "./about-tab";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { Button } from "@/components/ui/button";
import { twMerge } from "tailwind-merge";

export type SettingsTab = "general" | "privacy" | "about";

export function SettingsPage() {
  const { tab } = useSearch({ from: "/settings" });
  const sections: { key: SettingsTab; label: string }[] = [
    { key: "general", label: "General" },
    { key: "privacy", label: "Privacy" },
    { key: "about", label: "About Mirai" },
  ];

  const tabComponents: Record<SettingsTab, React.ReactNode> = {
    general: <GeneralTab />,
    privacy: <PrivacyTab />,
    about: <AboutTab />,
  };

  const isSidebarOpen = useSidebarStore((s) => s.isOpen);

  return (
    <div className="h-[calc(100vh-24px)] w-full bg-background text-label-title pt-4">
      <div className="h-full flex flex-col">
        <div className={twMerge("px-5 text-center lg:text-left sticky top-0 z-10 bg-background")}>
          <h1
            className={twMerge(
              "text-[15px] leading-[150%] font-medium mb-4 transition-all duration-300 ease-in-out",
              !isSidebarOpen && "text-center",
            )}
          >
            Settings
          </h1>
        </div>
        <div className="h-[1px] bg-cell-border" />

        <div className="lg:hidden px-5 py-4 sticky top-[52px] z-10 bg-background">
          <div className="flex gap-2 overflow-x-auto thin-scrollbar">
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
          <aside className="hidden lg:block w-[152px] flex-shrink-0 h-full pt-2 overflow-auto thin-scrollbar border-r border-cell-border">
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
              "flex-1 lg:px-0 h-full overflow-auto thin-scrollbar [scrollbar-gutter:stable]",
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
