import { getPlatform } from "@/platform/platform-singleton";
import { isDarkTheme, useAppStore } from "@/stores/use-app-store";
import { useEffect, useLayoutEffect } from "react";
import { useMediaQuery } from "./use-media-query";

export const useAppTheme = () => {
  const theme = useAppStore((state) => state.theme);
  const systemDark = useMediaQuery("(prefers-color-scheme: dark)");

  useLayoutEffect(() => {
    const isDarkMode = isDarkTheme(theme, systemDark);
    document.documentElement.classList.toggle("dark", isDarkMode);
    if (useAppStore.getState().isDarkMode !== isDarkMode) useAppStore.setState({ isDarkMode });
  }, [theme, systemDark]);

  useEffect(() => {
    void getPlatform()
      .systemUi.setWindowTheme(theme)
      .catch(() => {});
  }, [theme]);
};
