import { createFileRoute } from "@tanstack/react-router";
import { LocalModelsPage } from "@/features/local-models/components/local-models-page";

export const Route = createFileRoute("/local-models")({
  component: LocalModelsRoute,
});

function LocalModelsRoute() {
  return <LocalModelsPage />;
}
