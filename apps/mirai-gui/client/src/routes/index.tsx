import { Loader } from "@/components/loader";
import { platformInfo } from "@/platform/platform-info";
import { hasSeenWelcome } from "@/features/welcome/welcome-seen";
import { useModelsStore } from "@/stores/use-models-store";
import { modelDownloadPhases } from "@/types/model-manager";
import { createFileRoute, Navigate } from "@tanstack/react-router";
import { useState } from "react";
import { v4 as uuidv4 } from "uuid";

export const Route = createFileRoute("/")({
  component: HomePage,
});

function HomePage() {
  const [chatId] = useState(() => uuidv4());
  const models = useModelsStore((s) => s.models);
  const modelPhasesById = useModelsStore((s) => s.modelPhasesById);
  const initialized = useModelsStore((s) => s.initialized);
  const catalogComplete = useModelsStore((s) => s.catalogComplete);
  const catalogRefreshing = useModelsStore((s) => s.catalogRefreshing);
  const error = useModelsStore((s) => s.error);

  if (platformInfo.isWeb) return <Navigate to="/chats" replace />;

  // A new install says hello first.
  if (!hasSeenWelcome()) return <Navigate to="/welcome" replace />;

  if (models.some((model) => modelPhasesById[model.repoId] === modelDownloadPhases.downloaded)) {
    return <Navigate to="/chat/$chatId" params={{ chatId }} search={{ isNew: true }} replace />;
  }

  const checkingModels =
    !initialized ||
    (!error &&
      ((!catalogComplete && catalogRefreshing) ||
        models.some((model) => {
          const phase = modelPhasesById[model.repoId];
          return phase === undefined || phase === modelDownloadPhases.initializing;
        })));

  if (checkingModels) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader text="Checking available models…" />
      </div>
    );
  }

  return <Navigate to="/local-models" replace />;
}
