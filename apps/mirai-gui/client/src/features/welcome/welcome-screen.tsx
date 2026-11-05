import { Field, Label } from "@headlessui/react";
import { IconCircleInfo } from "@/components/icons/app-icons";
import { useNavigate } from "@tanstack/react-router";
import { type CSSProperties, useEffect, useState } from "react";
import { twMerge } from "tailwind-merge";
import { UzuCat } from "@/components/uzu-cat";
import { Checkbox } from "@/components/ui/checkbox";
import { Tooltip } from "@/components/ui/tooltip";
import { useMediaQuery } from "@/hooks/use-media-query";
import { useSettingsStore } from "@/stores/use-settings-store";
import { markWelcomeSeen } from "./welcome-seen";

// Each part comes into focus a beat after the one above it, out of a deep blur: a screen seen once
// can take its time (about 1.4 s in all). Under reduced motion the entrance is off.
const STEP_MS = 120;
const arrive = (step: number): CSSProperties => ({ animationDelay: `${step * STEP_MS}ms` });
const LEAVE_MS = 260;

/**
 * The first thing a new install shows, before the app itself: the cat says hello, what Mirai is
 * in one line, and the way in. Usage data sharing is the real setting, so it can be
 * decided here or later in Settings.
 */
export function WelcomeScreen() {
  const navigate = useNavigate();
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const analyticsEnabled = useSettingsStore((s) => s.analyticsEnabled);
  const setAnalyticsEnabled = useSettingsStore((s) => s.setAnalyticsEnabled);
  const fetchSettings = useSettingsStore((s) => s.fetch);
  const [eager, setEager] = useState(false);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    void fetchSettings();
  }, [fetchSettings]);

  const getStarted = () => {
    if (leaving) return;
    markWelcomeSeen();
    if (reducedMotion) {
      void navigate({ to: "/", replace: true });
      return;
    }
    setLeaving(true);
    window.setTimeout(() => void navigate({ to: "/", replace: true }), LEAVE_MS);
  };

  return (
    <div
      data-tauri-drag-region
      className="flex h-full w-full items-center justify-center overflow-y-auto bg-background px-6 py-16"
    >
      <div
        className={twMerge(
          "flex w-full max-w-[520px] flex-col items-center text-center transition-[opacity,scale,filter] ease-[cubic-bezier(0.32,0.72,0,1)]",
          leaving && "scale-[0.98] opacity-0 blur-[6px]",
        )}
        style={{ transitionDuration: `${LEAVE_MS}ms` }}
      >
        <div className="mb-4 animate-uzu-welcome-cat" style={arrive(0)}>
          <UzuCat
            state={eager ? "ready" : "welcome"}
            size={128}
            className="text-text-primary"
            label="uzu, saying hello"
          />
        </div>

        <h1
          className="text-[32px] font-[620] leading-[1.1] tracking-[-0.025em] text-text-primary text-balance animate-uzu-welcome"
          style={arrive(1)}
        >
          Welcome to Mirai
        </h1>
        <p
          className="mt-3 text-[16px] leading-[1.5] text-text-secondary text-pretty animate-uzu-welcome"
          style={arrive(2)}
        >
          Try the full private and local AI potential of your Mac
        </p>

        <button
          type="button"
          // The first thing to press: Enter goes straight in.
          autoFocus
          onClick={getStarted}
          // Only a pointer makes the cat eager: the button has focus from the start, and switching the
          // cat's pose then would break its greeting mid-entrance.
          onPointerEnter={() => setEager(true)}
          onPointerLeave={() => setEager(false)}
          className="mt-10 h-11 min-w-[220px] rounded-full bg-primary px-7 text-[15px] font-[500] text-primary-contrast outline-hidden transition-[background-color,scale] duration-150 ease-out hover:bg-primary-hover active:scale-[0.96] focus-visible:shadow-focus animate-uzu-welcome"
          style={arrive(3)}
        >
          Get started
        </button>

        <Field className="mt-5 flex items-center gap-2 animate-uzu-welcome" style={arrive(4)}>
          <Checkbox checked={analyticsEnabled} onChange={(checked) => void setAnalyticsEnabled(checked)} />
          <Label className="cursor-pointer select-none text-[13px] text-text-muted">Share anonymous usage data</Label>
          <Tooltip
            content="Model activity, performance measurements, OS/CPU/RAM, app versions, and a temporary session ID. You can change this later in Settings."
            side="top"
          >
            <span
              tabIndex={0}
              aria-label="What is shared"
              className="flex size-5 cursor-help items-center justify-center rounded-full text-text-muted outline-hidden transition-colors duration-150 ease-out hover:text-text-primary focus-visible:shadow-focus"
            >
              <IconCircleInfo size={16} aria-hidden />
            </span>
          </Tooltip>
        </Field>
      </div>
    </div>
  );
}
