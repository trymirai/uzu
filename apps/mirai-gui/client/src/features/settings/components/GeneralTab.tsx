import { GlobalInstructions, Toggle } from "@/components/ui";
import { platformInfo } from "@/platform/platformInfo";
import { useAppStore } from "@/stores/useAppStore";
import { useGlobalInstructionsStore } from "@/stores/useGlobalInstructionsStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useEffect } from "react";
import { AutoEjectSetting } from "./general/AutoEjectSetting";
import { CliInstallSetting } from "./general/CliInstallSetting";
import { FeedbackFooter } from "./general/FeedbackFooter";
import { QuickEntryShortcutSetting } from "./general/QuickEntryShortcutSetting";
import { SettingDivider, SettingRow } from "./general/SettingRow";

export default function GeneralTab() {
  const {
    enableThinking,
    runOnStartup,
    quickEntryShortcut,
    autoEjectEnabled,
    autoEjectMinutes,
    fetch: fetchSettings,
    fetchDesktopSettings,
    setEnableThinking,
    setRunOnStartup,
    registerQuickEntryShortcut,
    unregisterQuickEntryShortcut,
    setAutoEjectEnabled,
    setAutoEjectMinutes,
  } = useSettingsStore();

  const instructions = useGlobalInstructionsStore((s) => s.instructions);
  const loadInstructions = useGlobalInstructionsStore((s) => s.loadInstructions);
  const saveInstructions = useGlobalInstructionsStore((s) => s.saveInstructions);

  const isDarkMode = useAppStore((s) => s.isDarkMode);
  const setDarkMode = useAppStore((s) => s.setDarkMode);

  useEffect(() => {
    loadInstructions();
    fetchSettings();
    fetchDesktopSettings().catch(() => {});
  }, [loadInstructions, fetchSettings, fetchDesktopSettings]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-col gap-5 pb-5">
        <div className="flex flex-col gap-2 px-5 lg:max-w-[800px] mx-auto w-full">
          <GlobalInstructions instructions={instructions} onSave={saveInstructions} />
        </div>

        <SettingDivider />
        <SettingRow
          title="Dark mode"
          description="Use the dark appearance across the app"
          control={<Toggle checked={isDarkMode} onChange={() => setDarkMode(!isDarkMode)} />}
        />

        {platformInfo.features.startupLaunch && (
          <>
            <SettingDivider />
            <SettingRow
              title="Run on startup"
              description="Automatically start Mirai when you log in to your computer"
              control={<Toggle checked={runOnStartup} onChange={() => setRunOnStartup(!runOnStartup)} />}
            />
          </>
        )}

        {platformInfo.features.globalShortcut && (
          <>
            <SettingDivider />
            <QuickEntryShortcutSetting
              quickEntryShortcut={quickEntryShortcut}
              registerQuickEntryShortcut={registerQuickEntryShortcut}
              unregisterQuickEntryShortcut={unregisterQuickEntryShortcut}
            />
          </>
        )}

        <CliInstallSetting />

        <SettingDivider />
        <SettingRow
          title="Default reasoning mode"
          description="Let reasoning models think by default. You can override this for each model."
          control={<Toggle checked={enableThinking} onChange={() => setEnableThinking(!enableThinking)} />}
        />

        {platformInfo.features.autoEject && (
          <>
            <SettingDivider />
            <AutoEjectSetting
              autoEjectEnabled={autoEjectEnabled}
              autoEjectMinutes={autoEjectMinutes}
              setAutoEjectEnabled={setAutoEjectEnabled}
              setAutoEjectMinutes={setAutoEjectMinutes}
            />
          </>
        )}
      </div>

      <FeedbackFooter />
    </div>
  );
}
