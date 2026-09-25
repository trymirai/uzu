import { ErrorPage } from "@/components/error-page";
import { FooterBar } from "@/components/footer-bar";
import { useAppNavigationEvents } from "@/hooks/useAppNavigationEvents";
import { useIsMobile } from "@/hooks/useMediaQuery";
import { useGlobalDownloadToasts } from "@/hooks/useGlobalDownloadToasts";
import { platformInfo } from "@/platform/platformInfo";
import { useSidebarStore } from "@/stores/useSidebarStore";
import { createRootRoute, Outlet, useLocation } from "@tanstack/react-router";
import { useEffect } from "react";
import { twMerge } from "tailwind-merge";
import LeftBar from "../components/left-bar";
import { ToastProvider } from "@/ui-kit";
import { useAppInitialization } from "../hooks/useAppInitialization";

export const Route = createRootRoute({
  component: RootComponent,
  notFoundComponent: () => <ErrorPage />,
});

function RootComponent() {
  const location = useLocation();
  const isWelcome = location.pathname === "/welcome";
  const isMobile = useIsMobile();
  const setMobile = useSidebarStore((s) => s.setMobile);

  useAppInitialization(!isWelcome);
  useGlobalDownloadToasts();
  useAppNavigationEvents();

  useEffect(() => {
    setMobile(isMobile);
  }, [isMobile, setMobile]);

  if (location.pathname === "/welcome") {
    return (
      <div className="min-h-screen overflow-hidden">
        <div
          data-tauri-drag-region
          style={platformInfo.isTauri ? { pointerEvents: "auto" } : undefined}
          className="drag-layer fixed top-0 left-0 right-0 h-10 select-none"
        />
        <Outlet />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg dark:bg-bg-dark text-label-title dark:text-label-title-dark flex thin-scrollbar">
      <div className="drag-layer fixed top-0 left-0 right-0 h-10 select-none" />
      <LeftBar />
      <div className={twMerge("relative flex-1 flex flex-col h-[100dvh] overflow-hidden", isMobile ? "w-full" : "")}>
        <main className="flex-1 overflow-auto thin-scrollbar">
          <div>
            <Outlet />
          </div>
        </main>
        <FooterBar />
      </div>
      <ToastProvider />
    </div>
  );
}
