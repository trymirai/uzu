import { useToast } from "@/components/ui/toast/use-toast";
import { getPlatform } from "@/platform/platform-singleton";
import { platformInfo } from "@/platform/platform-info";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { SettingDivider, SettingRow } from "./setting-row";

export function CliInstallSetting() {
  const [isInstalling, setIsInstalling] = useState(false);
  const [cliMissing, setCliMissing] = useState(false);
  const toast = useToast();

  useEffect(() => {
    if (!platformInfo.features.cliInstall) return;
    void getPlatform()
      .system.getCliStatus()
      .then((status) => setCliMissing(status === "missing"));
  }, []);

  const install = async () => {
    if (isInstalling) return;
    setIsInstalling(true);
    try {
      const result = await getPlatform().system.installCli();
      if (result === "installed" || result === "already-installed") {
        toast.success("The mirai command is now available in your terminal");
        setCliMissing(false);
      }
    } catch (error) {
      if (!(error instanceof Error && error.message === "cancelled")) {
        toast.error("Failed to install the CLI");
        console.error(error);
      }
    } finally {
      setIsInstalling(false);
    }
  };

  if (!platformInfo.features.cliInstall || !cliMissing) return null;

  return (
    <>
      <SettingDivider />
      <SettingRow
        title="Command line tool"
        description="Install the mirai command to use Mirai from the terminal"
        control={
          <Button
            kind="primary"
            size="xs"
            loading={isInstalling}
            disabled={isInstalling}
            onClick={() => {
              void install();
            }}
          >
            {isInstalling ? "Installing…" : "Install"}
          </Button>
        }
      />
    </>
  );
}
