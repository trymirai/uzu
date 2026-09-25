import { Link } from "@tanstack/react-router";
import Socials from "./socials";

function Settings() {
  return (
    <div>
      <div className="border-t border-cell-border dark:border-cell-border-dark mb-2" />
      <div className="flex flex-col gap-2">
        <Link
          to="/settings"
          search={{ tab: "general" }}
          className="flex px-2"
          activeOptions={{ includeSearch: true, exact: true }}
        >
          <div className="flex items-center gap-2 px-2 py-[6px] w-full rounded-md hover:bg-bg-hover hover:dark:bg-bg-hover-dark">
            <span className="text-[13px] font-[350] leading-[150%] text-label-title dark:text-label-title-dark">
              Settings
            </span>
          </div>
        </Link>
        <div className="flex px-2">
          <div className="px-2 py-[6px]">
            <Socials />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
