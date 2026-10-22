import { Link } from "@tanstack/react-router";
import { useSidebarStore } from "@/stores/use-sidebar-store";
import { useIsMobile } from "@/hooks/use-media-query";

type MenuItemProps = {
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
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
        className={`flex items-center gap-2 px-2 py-[6px] w-full rounded-md transition-colors duration-150 ${isActive ? "bg-sidebar-chat-selected" : "hover:bg-sidebar-chat-hover"}`}
      >
        <Icon className="w-5 h-5 flex-shrink-0 text-label-muted" />
        <p className="text-[13px] font-[350] leading-[150%] text-label-title truncate min-w-0">{title}</p>
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
