import { useEffect, useState } from "react";
import { Logo } from "@/components/icons/logo";
import { LogoLightMode } from "@/components/icons/logo-light-mode";
import { Button } from "@/components/ui/button";

export function WelcomeScreen() {
  const [isDarkMode, setIsDarkMode] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const updateTheme = () => setIsDarkMode(media.matches);
    updateTheme();
    media.addEventListener("change", updateTheme);
    return () => media.removeEventListener("change", updateTheme);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode);
  }, [isDarkMode]);

  const getStarted = (
    <Button kind="primary" size="lg" fullWidth disabled>
      Get started
    </Button>
  );

  return (
    <main className="h-dvh overflow-y-auto bg-background text-label-title">
      <div className="relative flex min-h-full justify-center">
        <div className="grid w-full max-w-[500px] grid-rows-[auto_1fr_auto] px-6 pt-10 pb-8 md:flex md:min-h-0 md:flex-col md:justify-center md:px-8 md:pt-12 md:pb-10">
          <div className="flex md:justify-center" aria-hidden="true">
            {isDarkMode ? (
              <Logo width={66} height={57} className="text-label-title" />
            ) : (
              <LogoLightMode width={48} height={24} />
            )}
          </div>

          <div className="flex flex-col justify-center md:mt-[162px] md:items-center md:text-center">
            <h1 className="text-[28px] leading-[130%] font-medium text-label-title">Welcome to Mirai</h1>
            <p className="mt-4 text-[16px] leading-[130%] font-[450] text-label-muted md:mt-3">
              Private AI, running locally in your browser
            </p>
            <div className="mt-8 w-full md:hidden">{getStarted}</div>
          </div>

          <div className="flex w-full flex-col items-start md:mt-[120px] md:items-center">
            <div className="mt-8 hidden w-full md:block">{getStarted}</div>
          </div>
        </div>
      </div>
    </main>
  );
}
