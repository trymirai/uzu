import { platformInfo } from "@/platform/platformInfo";
import { createFileRoute, Navigate } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: HomePage,
});

function HomePage() {
  return <Navigate to={platformInfo.isWeb ? "/chats" : "/local-models"} />;
}
