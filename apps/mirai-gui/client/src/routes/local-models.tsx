import { createFileRoute } from "@tanstack/react-router";
import { LocalModelsPage } from "@/features/local-models/components/local-models-page";

export const Route = createFileRoute("/local-models")({
  component: LocalModelsRoute,
  validateSearch: (search: Record<string, unknown>): { family?: string } =>
    typeof search.family === "string" ? { family: search.family } : {},
});

function LocalModelsRoute() {
  return <LocalModelsPage />;
}
