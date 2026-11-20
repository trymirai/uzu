import { platformInfo } from "@/platform/platform-info";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import type { ReactElement } from "react";

// Clear of the sidebar button, which in the Mac app sits 90 screen points in at every zoom.
const HEADER_LEFT_RESERVE = platformInfo.features.nativeTitleBar
  ? "calc(90px / var(--app-zoom, 1) + 30px)"
  : platformInfo.features.titleBar
    ? "0px"
    : "56px";

export const PageHeader = ({ title }: { title: ReactElement }) => {
  const isSidebarOpen = useSidebarStore((s) => s.isOpen);

  return (
    <div
      className="transition-[padding-left] duration-300 ease-in-out motion-reduce:transition-none"
      style={{ paddingLeft: isSidebarOpen ? 0 : HEADER_LEFT_RESERVE }}
    >
      {title}
    </div>
  );
};
