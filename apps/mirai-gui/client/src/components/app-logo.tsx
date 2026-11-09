import { useAppStore } from "@/stores/use-app-store";
import { Logo } from "./icons/logo";
import { LogoLightMode } from "./icons/logo-light-mode";

export function AppLogo() {
  const isDarkMode = useAppStore((s) => s.isDarkMode);
  return isDarkMode ? <Logo width={66} height={57} /> : <LogoLightMode width={48} height={24} />;
}
