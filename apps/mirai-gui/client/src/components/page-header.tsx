import { platformInfo } from "@/platform/platform-info";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import type { ReactElement } from "react";

const HEADER_LEFT_RESERVE_PX = platformInfo.features.nativeTitleBar ? 120 : 56;

export const PageHeader = ({ title }: { title: ReactElement }) => {
  const isSidebarOpen = useSidebarStore((s) => s.isOpen);

  return (
    <div
      className="transition-[padding-left] duration-300 ease-in-out motion-reduce:transition-none"
      style={{ paddingLeft: isSidebarOpen ? 0 : HEADER_LEFT_RESERVE_PX }}
    >
      {title}
    </div>
  );
};
