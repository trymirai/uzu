import { Button } from "@/components/ui/button";
import { Heart } from "lucide-react";
import { SettingDivider } from "./setting-row";

export function FeedbackFooter() {
  return (
    <div className="mt-auto sticky bottom-0 z-10">
      <SettingDivider />
      <div className="bg-bg/80 dark:bg-bg-dark/80 backdrop-blur flex items-center justify-between px-5 py-3 lg:max-w-[800px] mx-auto w-full gap-3">
        <div className="flex items-center gap-2 text-[13px] text-label-muted dark:text-label-muted-dark max-w-[210px] lg:max-w-none">
          <Heart className="min-w-4 min-h-4 text-blue" />
          <span>Let us know your feedback or request a new feature</span>
        </div>
        <Button kind="primary" size="xs" href="https://discord.gg/trymirai" target="_blank">
          Give Feedback
        </Button>
      </div>
    </div>
  );
}
