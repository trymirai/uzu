import { ChevronRight, Download, Shield } from "lucide-react";
import { Button, useToast } from "@/ui-kit";
import { useEffect, useState } from "react";
import { ClearDataDialog } from "./ClearDataDialog";
import { FeedbackFooter } from "./general/FeedbackFooter";
import { SettingDivider, SettingRow } from "./general/SettingRow";
import { platformInfo } from "@/platform/platformInfo";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { exportAllChatsZip } from "@/features/chat/services/exportChats";

const links = [
  {
    title: "Terms of Service",
    url: "https://artifacts.trymirai.com/legal/Mirai_Tech_Terms_of_Use.pdf",
  },
  {
    title: "Privacy Policy",
    url: "https://artifacts.trymirai.com/legal/Mirai_Tech_Privacy_Policy.pdf",
  },
];

export default function PrivacyTab() {
  const [isExporting, setIsExporting] = useState(false);
  const [isExportingLogs, setIsExportingLogs] = useState(false);
  const [clearDataOpen, setClearDataOpen] = useState(false);
  const toast = useToast();
  const { fetch: fetchSettings, exportLogs } = useSettingsStore();

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  const handleExportChats = async () => {
    if (isExporting) return;
    setIsExporting(true);
    try {
      const ok = await exportAllChatsZip();
      if (ok) toast.success("Exported chats to ZIP");
      else toast.info("Nothing to export");
    } catch {
      toast.error("Export failed");
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportLogs = async () => {
    if (isExportingLogs) return;
    setIsExportingLogs(true);
    const result = await exportLogs();
    setIsExportingLogs(false);
    if (result === "ok") toast.success("Exported logs");
    else if (result === "error") toast.error("Failed to export logs");
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-col">
        <div className="flex flex-col gap-4 px-5 lg:max-w-[800px] mx-auto w-full">
          <Shield className="w-6 h-6 text-blue" />
          <div className="flex flex-col gap-1">
            <h3 className="text-[18px] font-medium leading-[130%] tracking-[0.2px] text-label-title dark:text-label-title-dark break-words">
              All data is processed and stored locally on your device.
            </h3>
            <p className="text-[13px] font-[350] leading-[150%] text-label-muted dark:text-label-muted-dark">
              Your privacy is built into the core of how Mirai runs.
            </p>
          </div>
        </div>
        <div className="mt-5">
          <SettingDivider />
        </div>

        <div className="flex flex-col">
          {links.map((link, index) => (
            <div key={link.url} className="w-full">
              <a
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="lg:max-w-[800px] mx-auto w-full hover:bg-bg-sub hover:dark:bg-bg-sub-dark flex items-center justify-between px-5 py-5"
              >
                <h4 className="text-[15px] font-[350] leading-[150%] text-label-title dark:text-label-title-dark">
                  {link.title}
                </h4>

                <ChevronRight className="w-[14px] h-[14px] text-label-muted dark:text-label-muted-dark" />
              </a>
              {index < links.length - 1 && <SettingDivider />}
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-5">
          <SettingDivider />
          <SettingRow
            title="Export all your chats"
            description="As Markdown files in .zip archive"
            control={
              <Button
                kind="primary"
                size="sm"
                loading={isExporting}
                icon={Download}
                onClick={handleExportChats}
                disabled={isExporting}
              >
                Export
              </Button>
            }
          />
          {platformInfo.features.logExport && (
            <>
              <SettingDivider />
              <SettingRow
                title="Export logs"
                description="Save the current Mirai log file for debugging"
                control={
                  <Button
                    kind="primary"
                    size="sm"
                    loading={isExportingLogs}
                    icon={Download}
                    onClick={handleExportLogs}
                    disabled={isExportingLogs}
                  >
                    Export
                  </Button>
                }
              />
            </>
          )}
          <SettingDivider />
          <SettingRow
            title="Clear data"
            description="Delete dialogs, models, or logs"
            control={
              <Button kind="secondary" size="sm" onClick={() => setClearDataOpen(true)}>
                Clear data
              </Button>
            }
          />
          <SettingDivider />
        </div>
      </div>
      <ClearDataDialog open={clearDataOpen} onClose={() => setClearDataOpen(false)} />

      <FeedbackFooter />
    </div>
  );
}
