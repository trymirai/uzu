import { Link } from "@tanstack/react-router";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useIsMobile } from "@/hooks/use-media-query";
import { useState } from "react";

type MenuItemProps = {
  icon: React.ComponentType<{ className?: string; size?: number; active?: boolean }>;
  title: string;
  url?: string;
  onClick?: () => void;
  isActive?: boolean;
};

function MenuItem({ icon: Icon, title, url, onClick, isActive }: MenuItemProps) {
  const isMobile = useIsMobile();
  const closeOnMobile = useSidebarStore((s) => s.closeOnMobile);
  // Mouse hover only: the icon's small motion is feedback for a pointer, not for keyboard focus.
  const [hovered, setHovered] = useState(false);

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else if (url && isMobile) {
      closeOnMobile();
    }
  };

  const content = (
    <div
      className="flex px-2"
      onPointerEnter={(e) => e.pointerType === "mouse" && setHovered(true)}
      onPointerLeave={() => setHovered(false)}
    >
      <div
        className={`flex h-9 items-center gap-2.5 px-2.5 w-full rounded-[10px] transition-colors duration-150 ease-out ${isActive ? "bg-sidebar-chat-selected" : "hover:bg-sidebar-chat-hover"}`}
      >
        <Icon size={20} active={hovered} className="flex-shrink-0 text-label-title" />
        <p className="text-[14px] font-[450] leading-[1.3] text-label-title truncate min-w-0">{title}</p>
      </div>
    </div>
  );

  if (onClick) {
    return (
      <button onClick={handleClick} className="w-full text-left">
        {content}
      </button>
    );
  }

  return (
    <Link to={url || "#"} onClick={handleClick}>
      {content}
    </Link>
  );
}

export default MenuItem;
