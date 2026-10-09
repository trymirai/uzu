import { Download } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast/use-toast";
import { exportAllChatsZip } from "@/features/chat/services/export-chats";
import { platformInfo } from "@/platform/platform-info";
import { useSettingsStore } from "@/stores/use-settings-store";
import { ClearDataDialog } from "../clear-data-dialog";
import { SettingDivider, SettingRow } from "./setting-row";

export function DataSettings() {
  const [isExporting, setIsExporting] = useState(false);
  const [isExportingLogs, setIsExportingLogs] = useState(false);
  const [clearDataOpen, setClearDataOpen] = useState(false);
  const toast = useToast();
  const { exportLogs } = useSettingsStore();

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
    <>
      <SettingDivider />
      <SettingRow
        title="Export all your chats"
        description="As Markdown files in a ZIP archive"
        control={
          <Button
            kind="primary"
            size="sm"
            loading={isExporting}
            icon={<Download size={16} />}
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
                icon={<Download size={16} />}
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
        description="Delete chats, models, or logs"
        control={
          <Button kind="secondary" size="sm" onClick={() => setClearDataOpen(true)}>
            Clear data
          </Button>
        }
      />
      <ClearDataDialog open={clearDataOpen} onClose={() => setClearDataOpen(false)} />
    </>
  );
}
