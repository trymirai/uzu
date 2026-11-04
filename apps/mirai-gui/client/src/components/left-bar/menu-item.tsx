import { Link } from "@tanstack/react-router";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useIsMobile } from "@/hooks/use-media-query";

type MenuItemProps = {
  icon: React.ComponentType<{ className?: string; size?: number; "aria-hidden"?: boolean }>;
  title: string;
  url?: string;
  onClick?: () => void;
  isActive?: boolean;
};

function MenuItem({ icon: Icon, title, url, onClick, isActive }: MenuItemProps) {
  const isMobile = useIsMobile();
  const closeOnMobile = useSidebarStore((s) => s.closeOnMobile);

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else if (url && isMobile) {
      closeOnMobile();
    }
  };

  const content = (
    <div className="flex px-2">
      <div
        className={`flex h-9 items-center gap-2.5 px-2.5 w-full rounded-[10px] transition-colors duration-150 ease-out ${isActive ? "bg-sidebar-chat-selected" : "hover:bg-sidebar-chat-hover"}`}
      >
        <Icon aria-hidden size={20} className="flex-shrink-0 text-label-title" />
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
