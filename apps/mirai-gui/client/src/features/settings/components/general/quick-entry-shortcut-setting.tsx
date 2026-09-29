import { Button } from "@/components/ui/button";
import { X as XIcon } from "lucide-react";
import { SettingRow } from "./setting-row";
import { useShortcutCapture } from "./use-shortcut-capture";

export function QuickEntryShortcutSetting({
  quickEntryShortcut,
  registerQuickEntryShortcut,
  unregisterQuickEntryShortcut,
}: {
  quickEntryShortcut: string | null;
  registerQuickEntryShortcut: (accelerator: string) => Promise<boolean>;
  unregisterQuickEntryShortcut: () => Promise<void>;
}) {
  const { isCapturing, setIsCapturing, buttonRef, shortcutDisplay } = useShortcutCapture(
    quickEntryShortcut,
    registerQuickEntryShortcut,
  );
  const capturingRing = isCapturing ? "ring-2 ring-blue ring-offset-2 ring-offset-background" : "";

  return (
    <SettingRow
      title="Quick entry keyboard shortcut"
      description={isCapturing ? "Hold ⌘, ⌃ or ⌥, then a key" : "Open Mirai from anywhere with a shortcut"}
      control={
        quickEntryShortcut ? (
          <Button
            kind="primary"
            size="xs"
            icon={<XIcon width={16} height={16} />}
            iconPosition="right"
            className={capturingRing}
            ref={buttonRef}
            onClick={async () => {
              await unregisterQuickEntryShortcut();
              setIsCapturing(true);
            }}
          >
            {shortcutDisplay}
          </Button>
        ) : (
          <Button
            kind="primary"
            size="xs"
            className={capturingRing}
            ref={buttonRef}
            onClick={() => {
              setIsCapturing(true);
            }}
          >
            Set shortcut
          </Button>
        )
      }
    />
  );
}
