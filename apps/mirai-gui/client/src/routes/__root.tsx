import { ErrorPage } from "@/components/error-page";
import { FooterBar } from "@/components/footer-bar";
import { useAppNavigationEvents } from "@/hooks/use-app-navigation-events";
import { useSaveFailureToast } from "@/hooks/use-save-failure-toast";
import { useIsMobile } from "@/hooks/use-media-query";
import { useGlobalDownloadToasts } from "@/hooks/use-global-download-toasts";
import { platformInfo } from "@/platform/platform-info";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { createRootRoute, Outlet, useLocation } from "@tanstack/react-router";
import { useEffect } from "react";
import LeftBar from "@/components/left-bar";
import { ToastProvider } from "@/components/ui/toast/toast-provider";
import { useAppInitialization } from "@/hooks/use-app-initialization";

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
  useSaveFailureToast();

  useEffect(() => {
    setMobile(isMobile);
  }, [isMobile, setMobile]);

  if (location.pathname === "/welcome") {
    return (
      <div className="h-dvh overflow-clip">
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
    <div className="h-dvh overflow-clip bg-background text-label-title flex">
      <div className="drag-layer fixed top-0 left-0 right-0 h-10 select-none" />
      <LeftBar />
      <div className="relative flex-1 min-w-0 flex flex-col overflow-clip">
        <main className="flex-1 min-h-0 overflow-clip">
          <Outlet />
        </main>
        <FooterBar />
      </div>
      <ToastProvider />
    </div>
  );
}
