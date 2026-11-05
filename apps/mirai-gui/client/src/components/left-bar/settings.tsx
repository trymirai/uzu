import { Link } from "@tanstack/react-router";
import { useState } from "react";
import { SettingsIcon } from "./nav-icons";

/** Settings as a gear in the sidebar's bottom-left corner, lined up with the nav icons above. */
function Settings() {
  const [hovered, setHovered] = useState(false);
  return (
    <Link
      to="/settings"
      search={{ tab: "general" }}
      aria-label="Settings"
      activeOptions={{ includeSearch: false, exact: true }}
      onPointerEnter={(e) => e.pointerType === "mouse" && setHovered(true)}
      onPointerLeave={() => setHovered(false)}
      className="flex size-8 shrink-0 items-center justify-center rounded-[10px] text-label-title outline-hidden transition-colors duration-150 ease-out hover:bg-sidebar-chat-hover focus-visible:shadow-focus data-[status=active]:bg-sidebar-chat-selected"
    >
      <SettingsIcon size={20} active={hovered} />
    </Link>
  );
}

export default Settings;
