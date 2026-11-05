import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useModelsStore } from "@/stores/use-models-store";
import type { EngineModel } from "@/types/model-manager";
import { modelDownloadPhases } from "@/types/model-manager";
import { Route } from "./index";

const getModels = vi.hoisted(() => vi.fn());
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ models: { getModels } }) }));
vi.mock("@/platform/platform-info", () => ({ platformInfo: { isWeb: false } }));
vi.mock("@/components/loader", () => ({ Loader: ({ text }: { text: string }) => <div>{text}</div> }));
vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@tanstack/react-router")>()),
  Navigate: (props: object) => <div data-testid="destination">{JSON.stringify(props)}</div>,
}));

const HomePage = Route.options.component!;
const defaults = useModelsStore.getState();
const model = (phase: EngineModel["state"]["phase"]): EngineModel => ({
  identifier: "vendor/model",
  name: "Shared model",
  vendor: "Vendor",
  reasoning: { kind: "unsupported" },
  supportsTools: false,
  state: { phase, totalKbytes: 100, downloadedKbytes: 0, seq: 0 },
});
const destination = () => JSON.parse(screen.getByTestId("destination").textContent!);

beforeEach(() => {
  vi.clearAllMocks();
  useModelsStore.setState(defaults, true);
});
afterEach(cleanup);

it("waits for discovery and opens a new chat when a shared-cache download is found", async () => {
  render(<HomePage />);
  expect(screen.queryByTestId("destination")).toBeNull();

  getModels.mockResolvedValueOnce({ models: [], complete: false, refreshing: true });
  await act(() => useModelsStore.getState().fetchModels());
  expect(screen.queryByTestId("destination")).toBeNull();

  getModels.mockResolvedValueOnce({
    models: [model(modelDownloadPhases.initializing)],
    complete: true,
    refreshing: false,
  });
  await act(() => useModelsStore.getState().fetchModels());
  expect(screen.queryByTestId("destination")).toBeNull();

  act(() =>
    useModelsStore.getState().applyDownloadEvent({
      kind: "state",
      identifier: "vendor/model",
      phase: modelDownloadPhases.downloaded,
      completedBytes: 102400,
      totalBytes: 102400,
      error: null,
      seq: 1,
    }),
  );
  expect(destination()).toEqual({
    to: "/chat/$chatId",
    params: { chatId: expect.any(String) },
    search: { isNew: true },
    replace: true,
  });
});

it("opens a new chat from an installed model in the first snapshot without waiting for catalog refresh", async () => {
  getModels.mockResolvedValueOnce({
    models: [model(modelDownloadPhases.downloaded)],
    complete: false,
    refreshing: true,
  });
  render(<HomePage />);
  await act(() => useModelsStore.getState().fetchModels());
  expect(destination().to).toBe("/chat/$chatId");
});

it("opens Models only after pending downloads are confirmed absent", async () => {
  getModels.mockResolvedValueOnce({
    models: [model(modelDownloadPhases.initializing)],
    complete: true,
    refreshing: false,
  });
  render(<HomePage />);
  await act(() => useModelsStore.getState().fetchModels());
  expect(screen.queryByTestId("destination")).toBeNull();
  act(() =>
    useModelsStore.getState().applyDownloadEvent({
      kind: "state",
      identifier: "vendor/model",
      phase: modelDownloadPhases.notDownloaded,
      completedBytes: 0,
      totalBytes: 102400,
      error: null,
      seq: 1,
    }),
  );
  expect(destination()).toEqual({ to: "/local-models", replace: true });
});

it.each([true, false])("opens Models for an empty settled catalog (complete: %s)", async (complete) => {
  getModels.mockResolvedValueOnce({ models: [], complete, refreshing: false });
  render(<HomePage />);
  await act(() => useModelsStore.getState().fetchModels());
  expect(destination()).toEqual({ to: "/local-models", replace: true });
});

it("opens Models after discovery fails instead of waiting indefinitely", async () => {
  getModels.mockRejectedValueOnce(new Error("Catalog unavailable"));
  render(<HomePage />);
  await act(() => useModelsStore.getState().fetchModels());
  expect(destination()).toEqual({ to: "/local-models", replace: true });
});
