import { platformInfo } from "@/platform/platformInfo";
import { useSidebarStore } from "@/stores/useSidebarStore";
import type { ReactElement } from "react";

const HEADER_LEFT_RESERVE_PX = platformInfo.features.nativeTitleBar ? 120 : 56;

export const PageHeader = ({ title }: { title: ReactElement }) => {
  const isSidebarOpen = useSidebarStore((s) => s.isOpen);

  return (
    <div
      style={
        !isSidebarOpen
          ? {
              paddingLeft: `clamp(0px, calc(${HEADER_LEFT_RESERVE_PX}px - var(--sidebar-width, 0px)), ${HEADER_LEFT_RESERVE_PX}px)`,
            }
          : undefined
      }
    >
      {title}
    </div>
  );
};
