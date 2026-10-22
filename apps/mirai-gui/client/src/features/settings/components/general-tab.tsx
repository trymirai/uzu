import GlobalInstructions from "@/features/chat-history/components/global-instructions";
import { Toggle } from "@/components/ui/toggle";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { platformInfo } from "@/platform/platform-info";
import { useAppStore } from "@/stores/use-app-store";
import { useGlobalInstructionsStore } from "@/stores/use-global-instructions-store";
import { useSettingsStore } from "@/stores/use-settings-store";
import { useEffect } from "react";
import { AnalyticsSetting } from "./general/analytics-setting";
import { AutoEjectSetting } from "./general/auto-eject-setting";
import { CliInstallSetting } from "./general/cli-install-setting";
import { DataSettings } from "./general/data-settings";
import { SettingDivider, SettingRow } from "./general/setting-row";
import { ThemeSelector } from "./general/theme-selector";

export default function GeneralTab() {
  const {
    analyticsEnabled,
    modelChatNamingEnabled,
    autoEjectEnabled,
    autoEjectMinutes,
    fetch: fetchSettings,
    setAnalyticsEnabled,
    setModelChatNamingEnabled,
    setAutoEjectEnabled,
    setAutoEjectMinutes,
  } = useSettingsStore();

  const instructions = useGlobalInstructionsStore((s) => s.instructions);
  const loadInstructions = useGlobalInstructionsStore((s) => s.loadInstructions);
  const saveInstructions = useGlobalInstructionsStore((s) => s.saveInstructions);

  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  const chatWidth = useAppStore((s) => s.chatWidth);
  const setChatWidth = useAppStore((s) => s.setChatWidth);

  useEffect(() => {
    loadInstructions();
    fetchSettings();
  }, [loadInstructions, fetchSettings]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-col gap-5 pb-5">
        <div className="flex flex-col gap-2 px-5 lg:max-w-[800px] mx-auto w-full">
          <GlobalInstructions instructions={instructions} onSave={saveInstructions} />
        </div>

        <SettingDivider />
        <SettingRow
          title="Theme"
          description="Choose an appearance or follow your system settings"
          control={<ThemeSelector value={theme} onChange={setTheme} />}
        />

        <SettingDivider />
        <SettingRow
          title="Chat width"
          description="Maximum width of messages, the message box, and chat history. Adapts to fit smaller windows."
          control={
            <div className="shrink-0">
              <SegmentedControl
                ariaLabel="Chat width"
                value={String(chatWidth)}
                onChange={(value) => setChatWidth(Number(value))}
                options={[
                  { value: "800", label: "Narrow" },
                  { value: "1000", label: "Medium" },
                  { value: "1200", label: "Wide" },
                ]}
              />
            </div>
          }
        />

        <CliInstallSetting />

        <SettingDivider />
        <SettingRow
          title="Let models name chats via tool call"
          description="Enabled by default for supported models with 2B or more parameters. Otherwise, the same model is asked to name the chat separately."
          control={
            <Toggle
              label="Let models name chats via tool call"
              checked={modelChatNamingEnabled}
              onChange={setModelChatNamingEnabled}
            />
          }
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

        <SettingDivider />
        <AnalyticsSetting enabled={analyticsEnabled} onChange={setAnalyticsEnabled} />

        <DataSettings />
      </div>
    </div>
  );
}
