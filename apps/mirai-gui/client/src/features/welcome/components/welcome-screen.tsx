import { platformInfo } from "@/platform/platform-info";
import { useNavigate } from "@tanstack/react-router";
import { cubicBezier, motion } from "motion/react";
import { Lock, WifiOff } from "lucide-react";
import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";
import { useAppStore } from "@/stores/use-app-store";

import { LightningIcon } from "@/components/icons/lightning-icon";
import { Logo } from "@/components/icons/logo";
import { LogoLightMode } from "@/components/icons/logo-light-mode";
import { Button } from "@/components/ui/button";

const ease = cubicBezier(0.22, 1, 0.36, 1);

function WelcomeScreen() {
  const completeWelcome = useAppStore((s) => s.completeWelcome);
  const isDarkMode = useAppStore((s) => s.isDarkMode);
  const navigate = useNavigate();

  const handleContinue = () => {
    completeWelcome();
    navigate({ to: platformInfo.isTauri ? "/local-models" : "/chats" });
  };

  const featureItem = (icon: ReactNode, text: string, delayS: number) => (
    <motion.div
      className="flex items-center gap-2"
      initial={{ y: 10, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 1.5, ease, delay: delayS }}
    >
      <div className="text-label-title">{icon}</div>
      <span className="text-[13px] leading-[130%] font-normal text-label-muted">{text}</span>
    </motion.div>
  );

  const renderAnimatedTitle = (text: string, baseDelay = 0.5, charDelay = 0.05) => {
    const chars = Array.from(text);

    return (
      <h1 className="text-[28px] leading-[130%] font-medium text-label-title">
        {chars.map((ch, i) => (
          <motion.span
            key={`${i}-${ch}`}
            style={{ display: "inline-block" }}
            initial={{
              opacity: 0,
              y: 40,
              scale: 2,
              filter: "blur(30px)",
            }}
            animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
            transition={{
              ease,
              duration: 1.5,
              delay: baseDelay + i * charDelay,
            }}
          >
            {ch === " " ? "\u00A0" : ch}
          </motion.span>
        ))}
      </h1>
    );
  };

  const renderAnimatedSubtitle = (lines: string[], baseDelay = 1, lineDelay = 0) => (
    <div className="mt-4 md:mt-3 text-[16px] leading-[130%] font-[450] text-label-muted">
      {lines.map((line, i) => (
        <motion.div
          key={`${i}-${line}`}
          initial={{
            opacity: 0,
            scale: 0.5,
            filter: "blur(50px)",
          }}
          animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
          transition={{
            ease,
            duration: 1.5,
            delay: baseDelay + i * lineDelay,
          }}
        >
          {line}
        </motion.div>
      ))}
    </div>
  );

  return (
    <div className={twMerge("relative max-h-screen flex justify-center h-full", !isDarkMode && "bg-background")}>
      <div className="w-full max-w-[500px] md:min-h-0 px-6 md:px-8 pt-10 md:pt-12 pb-8 md:pb-10 grid grid-rows-[auto_1fr_auto] md:flex md:flex-col md:justify-center">
        <div className="contents">
          <motion.div
            className="flex md:justify-center"
            initial={{ scale: 2, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 1.5, ease, delay: 0 }}
          >
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
          </motion.div>

          <div className="flex flex-col md:items-center justify-center md:text-center md:mt-[162px]">
            {renderAnimatedTitle("Welcome to Mirai", 0.5, 0.05)}
            {renderAnimatedSubtitle(
              [
                platformInfo.isWeb
                  ? "Your private AI assistant, accessible anywhere"
                  : "Try the full private and local AI potential of your Mac",
              ],
              1,
              0,
            )}
          </div>

          <div className="flex flex-col items-start md:items-center w-full md:mt-[120px]">
            <div className="flex flex-col md:flex-row md:items-center md:justify-center md:gap-6 gap-4">
              {featureItem(<Lock size={14} />, "100% Private", 2.0)}
              {featureItem(<LightningIcon className="w-[14px] h-[15px]" />, "Blazing Fast Responses", 2.2)}
              {!platformInfo.isWeb && featureItem(<WifiOff size={14} />, "Works Offline", 2.4)}
            </div>

            <div className="w-full mt-8">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 2, ease, delay: 3.2 }}
              >
                <Button kind="primary" size="lg" fullWidth onClick={handleContinue}>
                  Get started
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default WelcomeScreen;
