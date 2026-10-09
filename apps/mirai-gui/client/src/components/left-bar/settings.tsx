import { Link } from "@tanstack/react-router";
import { Settings as SettingsIcon } from "lucide-react";

function Settings() {
  return (
    <div>
      <div className="border-t border-cell-border mb-2" />
      <Link
        to="/settings"
        search={{ tab: "general" }}
        className="flex px-2"
        activeOptions={{ includeSearch: false, exact: true }}
      >
        {({ isActive }) => (
          <div
            className={`flex items-center gap-2 px-2 py-[6px] w-full rounded-md transition-colors duration-150 ${isActive ? "bg-sidebar-chat-selected" : "hover:bg-sidebar-chat-hover"}`}
          >
            <SettingsIcon aria-hidden="true" className="size-5 shrink-0 text-label-muted" />
            <span className="text-[13px] font-[350] leading-[150%] text-label-title">Settings</span>
          </div>
        )}
      </Link>
    </div>
  );
}

export default Settings;
