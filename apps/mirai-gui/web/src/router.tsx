import { Outlet, useLocation, useNavigate } from "@tanstack/react-router";
import { createAppRouter } from "@/router";
import { routeTree } from "@/route-tree.gen";
import { Route as indexRoute } from "@/routes/index";
import { RootComponent as SharedAppLayout } from "@/routes/__root";
import { useAppTheme } from "@/hooks/use-app-theme";
import { WelcomeScreen } from "./welcome-screen";

function WebRoot() {
  const isWelcome = useLocation({ select: (location) => location.pathname === "/" });
  return isWelcome ? <Outlet /> : <SharedAppLayout />;
}

function WelcomeRoute() {
  useAppTheme();
  const navigate = useNavigate();
  return <WelcomeScreen onContinue={() => void navigate({ to: "/local-models" })} />;
}

export function createWebRouter() {
  // These route instances belong to the web bundle. Reuse the complete shared
  // tree and layout, customizing only the browser's welcome entry point.
  routeTree.update({ component: WebRoot });
  indexRoute.update({ component: WelcomeRoute });
  return createAppRouter();
}
