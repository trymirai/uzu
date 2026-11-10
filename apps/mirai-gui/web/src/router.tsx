import { Navigate, Outlet, useLocation, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { createAppRouter } from "@/router";
import { routeTree } from "@/route-tree.gen";
import { Route as indexRoute } from "@/routes/index";
import { RootComponent as SharedAppLayout } from "@/routes/__root";
import { useAppTheme } from "@/hooks/use-app-theme";
import { readJson, writeJson } from "@/platform/services/shared/browser-store";
import { WelcomeScreen } from "./welcome-screen";

function WebRoot() {
  const isWelcome = useLocation({ select: (location) => location.pathname === "/" });
  return isWelcome ? <Outlet /> : <SharedAppLayout />;
}

// The welcome screen greets a visitor once; later visits open the models.
const WELCOMED_KEY = "mirai.web.welcomed";

function WelcomeRoute() {
  useAppTheme();
  const navigate = useNavigate();
  const [welcomed] = useState(() => readJson(WELCOMED_KEY, false));
  if (welcomed) return <Navigate to="/local-models" replace />;
  const onContinue = () => {
    try {
      writeJson(WELCOMED_KEY, true);
    } catch {
      // With storage blocked, the welcome screen simply appears again next time.
    }
    void navigate({ to: "/local-models" });
  };
  return <WelcomeScreen onContinue={onContinue} />;
}

export function createWebRouter() {
  // These route instances belong to the web bundle. Reuse the complete shared
  // tree and layout, customizing only the browser's welcome entry point.
  routeTree.update({ component: WebRoot });
  indexRoute.update({ component: WelcomeRoute });
  return createAppRouter();
}
