import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Link,
  Outlet,
  RouterProvider,
} from "@tanstack/react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Route as LocalModelsRoute } from "@/routes/local-models";
import { useModelsStore } from "@/stores/use-models-store";
import { ModelKind, type PlatformModel } from "@/types/models";
import { LocalModelsPage } from ".";

const model: PlatformModel = {
  repoId: "vendor/model",
  kind: ModelKind.Text,
  name: "Model",
  vendor: "Vendor",
  reasoning: { kind: "unsupported" },
  supportsTools: false,
  family: "Family",
  familyIdentifier: "family",
};
const modelsDefaults = useModelsStore.getState();

beforeEach(() => {
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} }));
  useModelsStore.setState({ ...modelsDefaults, models: [model], initialized: true, fetchModels: async () => {} }, true);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("leaves a family's models for the families when the sidebar opens Models", async () => {
  const root = createRootRoute({
    component: () => (
      <>
        <Link to="/local-models">Models</Link>
        <Outlet />
      </>
    ),
  });
  const page = createRoute({
    getParentRoute: () => root,
    path: "/local-models",
    validateSearch: LocalModelsRoute.options.validateSearch,
    component: LocalModelsPage,
  });
  const history = createMemoryHistory({ initialEntries: ["/local-models"] });
  render(<RouterProvider router={createRouter({ routeTree: root.addChildren([page]), history })} />);

  fireEvent.click(await screen.findByText("Family"));
  await screen.findByLabelText("Back to families");

  fireEvent.click(screen.getByText("Models"));
  await waitFor(() => expect(screen.queryByLabelText("Back to families")).toBeNull());
  expect(screen.getByText("Family")).toBeTruthy();
});
