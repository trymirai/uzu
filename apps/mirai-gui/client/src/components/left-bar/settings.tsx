import { Link } from "@tanstack/react-router";
import { IconSettingsGear2 } from "@central-icons-react/round-outlined-radius-3-stroke-2";

function Settings() {
  return (
    <div>
      <Link
        to="/settings"
        search={{ tab: "general" }}
        className="flex px-2"
        activeOptions={{ includeSearch: false, exact: true }}
      >
        {({ isActive }) => (
          <div
            className={`flex h-9 items-center gap-2.5 px-2.5 w-full rounded-[10px] transition-colors duration-150 ease-out ${isActive ? "bg-sidebar-chat-selected" : "hover:bg-sidebar-chat-hover"}`}
          >
            <IconSettingsGear2 aria-hidden size={20} className="shrink-0 text-label-title" />
            <span className="text-[14px] font-[450] leading-[1.3] text-label-title">Settings</span>
          </div>
        )}
      </Link>
    </div>
  );
}

export default Settings;
