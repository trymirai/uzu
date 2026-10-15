import { platformInfo } from "@/platform/platform-info";
import { useNavigate } from "@tanstack/react-router";
import { WifiOff } from "lucide-react";
import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { useAppStore } from "@/stores/use-app-store";

import { Logo } from "@/components/icons/logo";
import { LogoLightMode } from "@/components/icons/logo-light-mode";
import { Button } from "@/components/ui/button";

function WelcomeScreen() {
  const completeWelcome = useAppStore((s) => s.completeWelcome);
  const isDarkMode = useAppStore((s) => s.isDarkMode);
  const navigate = useNavigate();

  const handleContinue = () => {
    completeWelcome();
    navigate({ to: "/local-models" });
  };

  const featureItem = (icon: ReactNode, text: string) => (
    <div className="flex items-center gap-2">
      <div className="text-label-title">{icon}</div>
      <span className="text-[13px] leading-[130%] font-normal text-label-muted">{text}</span>
    </div>
  );

  const renderTitle = (text: string) => (
    <h1 className="text-[28px] leading-[130%] font-medium text-label-title">{text}</h1>
  );

  const renderSubtitle = (lines: string[]) => (
    <div className="mt-4 md:mt-3 text-[16px] leading-[130%] font-[450] text-label-muted">
      {lines.map((line, i) => (
        <div key={`${i}-${line}`}>{line}</div>
      ))}
    </div>
  );

  return (
    <div className={twMerge("relative max-h-screen flex justify-center h-full", !isDarkMode && "bg-background")}>
      <div className="w-full max-w-[500px] md:min-h-0 px-6 md:px-8 pt-10 md:pt-12 pb-8 md:pb-10 grid grid-rows-[auto_1fr_auto] md:flex md:flex-col md:justify-center">
        <div className="contents">
          <div className="flex md:justify-center">
            {isDarkMode ? (
              <Logo
                width={66}
                height={57}
                className="text-label-title"
                style={{
                  willChange: "transform,opacity",
                  transform: "translateZ(0)",
                }}
              />
            ) : (
              <LogoLightMode
                width={48}
                height={24}
                style={{
                  willChange: "transform,opacity",
                  transform: "translateZ(0)",
                }}
              />
            )}
          </div>

          <div className="flex flex-col md:items-center justify-center md:text-center md:mt-[162px]">
            {renderTitle("Welcome to Mirai")}
            {renderSubtitle([
              platformInfo.isWeb
                ? "Private AI, running locally in your browser"
                : "Try the full private and local AI potential of your Mac",
            ])}
            <div className="w-full mt-8 md:hidden">
              <Button kind="primary" size="lg" fullWidth onClick={handleContinue}>
                Get started
              </Button>
            </div>
          </div>

          <div className="flex flex-col items-start md:items-center w-full md:mt-[120px]">
            <div className="flex flex-col md:flex-row md:items-center md:justify-center md:gap-6 gap-4">
              {!platformInfo.isWeb && featureItem(<WifiOff size={14} />, "Works Offline")}
            </div>

            <div className="hidden md:block w-full mt-8">
              <div>
                <Button kind="primary" size="lg" fullWidth onClick={handleContinue}>
                  Get started
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default WelcomeScreen;
