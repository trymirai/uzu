import { ErrorPage } from "@/components/error-page";
import { useAppNavigationEvents } from "@/hooks/use-app-navigation-events";
import { useSaveFailureToast } from "@/hooks/use-save-failure-toast";
import { useIsMobile } from "@/hooks/use-media-query";
import { useGlobalDownloadToasts } from "@/hooks/use-global-download-toasts";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { createRootRoute, Outlet, useLocation } from "@tanstack/react-router";
import { useEffect } from "react";
import { twMerge } from "tailwind-merge";
import LeftBar from "@/components/left-bar";
import { ToastProvider } from "@/components/ui/toast/toast-provider";
import { TooltipProvider } from "@/components/ui/tooltip/provider";
import { useAppInitialization } from "@/hooks/use-app-initialization";

export const Route = createRootRoute({
  component: RootComponent,
  notFoundComponent: () => <ErrorPage />,
});

function RootComponent() {
  const isMobile = useIsMobile();
  const setMobile = useSidebarStore((s) => s.setMobile);
  const isSidebarOpen = useSidebarStore((s) => s.isOpen);
  const { pathname } = useLocation();

  useAppInitialization();
  useGlobalDownloadToasts();
  useAppNavigationEvents();
  useSaveFailureToast();

  useEffect(() => {
    setMobile(isMobile);
  }, [isMobile, setMobile]);

  // The welcome is its own full window, before the app: no sidebar, no panel.
  if (pathname === "/welcome") {
    return (
      <TooltipProvider>
        <div className="h-dvh overflow-clip bg-background text-label-title">
          <div className="drag-layer fixed top-0 left-0 right-0 h-10 select-none" />
          <Outlet />
          <ToastProvider />
        </div>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="h-dvh overflow-clip bg-bg-sidebar text-label-title flex">
        <div className="drag-layer fixed top-0 left-0 right-0 h-10 select-none" />
        <LeftBar />
        {/* Content sits in an inset, rounded panel on the window's own surface, like a sheet of
          paper on a desk: the panel's edge separates it from the sidebar, so neither needs a line. */}
        {/* The margin around the panel is window chrome: dragging it moves the window. */}
        <div
          data-tauri-drag-region
          className={twMerge(
            "relative flex-1 min-w-0 flex flex-col overflow-clip py-2 pr-2 transition-[padding] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none",
            isSidebarOpen && !isMobile ? "pl-0" : "pl-2",
          )}
        >
          <main className="flex-1 min-h-0 overflow-clip rounded-[12px] border-[0.5px] border-border-default bg-background shadow-[0_1px_2px_0_rgb(0_0_0/0.03)]">
            <Outlet />
          </main>
        </div>
        <ToastProvider />
      </div>
    </TooltipProvider>
  );
}
