import { createBrowserHistory, createRouter } from "@tanstack/react-router";
import { routeTree } from "./route-tree.gen";

export const createAppRouter = (initialPath: string) => {
  const history = createBrowserHistory();
  if (window.location.pathname === "/") history.replace(initialPath);
  return createRouter({ routeTree, history, notFoundMode: "root" });
};

export type AppRouter = ReturnType<typeof createAppRouter>;

declare module "@tanstack/react-router" {
  interface Register {
    router: AppRouter;
  }
}
