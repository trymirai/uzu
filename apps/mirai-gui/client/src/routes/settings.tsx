import { createFileRoute } from "@tanstack/react-router";
import { SettingsPage, type SettingsTab } from "@/features/settings/components/settings-page";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
  validateSearch: (search: Record<string, unknown>) => {
    const allowed = new Set<SettingsTab>(["general", "privacy", "about"]);
    const tab =
      typeof search.tab === "string" && allowed.has(search.tab as SettingsTab)
        ? (search.tab as SettingsTab)
        : "general";
    return { tab };
  },
});
