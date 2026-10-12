import { beforeEach, expect, it, vi } from "vitest";
import type { EngineModel, ModelCatalog } from "@/types/model-manager";
import { modelDownloadPhases } from "@/types/model-manager";
import { useModelsStore } from "./use-models-store";

const mocks = vi.hoisted(() => ({ getModels: vi.fn(), refreshModels: vi.fn() }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ models: mocks }) }));

const REPO_ID = "vendor/model";
const defaults = useModelsStore.getState();

const snapshot = (phase: EngineModel["state"]["phase"], seq: number): EngineModel => ({
  identifier: REPO_ID,
  repoId: REPO_ID,
  vendor: "Vendor",
  name: "Model",
  reasoning: { kind: "unsupported" },
  supportsTools: false,
  state: { phase, totalKbytes: 100, downloadedKbytes: 50, seq },
});

beforeEach(() => {
  vi.clearAllMocks();
  useModelsStore.setState(defaults, true);
});

it("keeps a download event that arrived while the catalog request was in flight", async () => {
  let resolveModels!: (value: ModelCatalog) => void;
  mocks.getModels.mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveModels = resolve;
      }),
  );
  const fetching = useModelsStore.getState().fetchModels();

  useModelsStore.getState().applyDownloadEvent({ kind: "done", identifier: REPO_ID, seq: 7 });
  resolveModels({ models: [snapshot(modelDownloadPhases.downloading, 3)], complete: true, refreshing: false });
  await fetching;

  expect(useModelsStore.getState().modelStatesById[REPO_ID]?.phase).toBe(modelDownloadPhases.downloaded);
});

it("applies a snapshot that is newer than the last event", async () => {
  useModelsStore.getState().applyDownloadEvent({ kind: "paused", identifier: REPO_ID, seq: 2 });
  mocks.getModels.mockResolvedValue({
    models: [snapshot(modelDownloadPhases.downloaded, 5)],
    complete: true,
    refreshing: false,
  });

  await useModelsStore.getState().fetchModels();

  expect(useModelsStore.getState().modelStatesById[REPO_ID]?.phase).toBe(modelDownloadPhases.downloaded);
});

it("ignores an event older than the state it already holds", () => {
  useModelsStore.getState().applyDownloadEvent({ kind: "done", identifier: REPO_ID, seq: 9 });
  useModelsStore.getState().applyDownloadEvent({ kind: "paused", identifier: REPO_ID, seq: 4 });

  expect(useModelsStore.getState().modelStatesById[REPO_ID]?.phase).toBe(modelDownloadPhases.downloaded);
});

it("refetches after catalog changes while a snapshot is in flight", async () => {
  let resolveModels!: (value: ModelCatalog) => void;
  mocks.getModels
    .mockImplementationOnce(
      () =>
        new Promise<ModelCatalog>((resolve) => {
          resolveModels = resolve;
        }),
    )
    .mockResolvedValueOnce({
      models: [snapshot(modelDownloadPhases.downloaded, 2)],
      complete: true,
      refreshing: false,
    });
  const fetching = useModelsStore.getState().fetchModels();
  void useModelsStore.getState().fetchModels();
  void useModelsStore.getState().fetchModels();
  resolveModels({ models: [], complete: false, refreshing: true });
  await fetching;

  expect(mocks.getModels).toHaveBeenCalledTimes(2);
  expect(useModelsStore.getState().catalogComplete).toBe(true);
  expect(useModelsStore.getState().models.map((model) => model.repoId)).toEqual([REPO_ID]);
});

it("settles errors and lets the next request retry", async () => {
  mocks.getModels.mockRejectedValueOnce(new Error("Catalog unavailable"));
  await useModelsStore.getState().fetchModels();
  expect(useModelsStore.getState()).toMatchObject({ initialized: true, loading: false, error: "Catalog unavailable" });

  mocks.getModels.mockResolvedValueOnce({ models: [], complete: true, refreshing: false });
  await useModelsStore.getState().fetchModels();
  expect(useModelsStore.getState()).toMatchObject({ hasLoadedModels: true, catalogComplete: true, error: null });
});

it("updates a pending model when its existing download is discovered", async () => {
  mocks.getModels.mockResolvedValueOnce({
    models: [snapshot(modelDownloadPhases.initializing, 0)],
    complete: false,
    refreshing: true,
  });
  await useModelsStore.getState().fetchModels();
  useModelsStore.getState().applyDownloadEvent({
    kind: "state",
    identifier: REPO_ID,
    seq: 1,
    phase: modelDownloadPhases.downloaded,
    completedBytes: 102400,
    totalBytes: 102400,
    error: null,
  });
  expect(useModelsStore.getState().modelStatesById[REPO_ID]).toMatchObject({
    phase: modelDownloadPhases.downloaded,
    downloadedKbytes: 100,
    totalKbytes: 100,
    seq: 1,
  });
});

it("distinguishes an incomplete failed catalog from a refresh in progress and explicitly retries it", async () => {
  mocks.getModels.mockResolvedValueOnce({ models: [], complete: false, refreshing: false });
  await useModelsStore.getState().fetchModels();
  expect(useModelsStore.getState()).toMatchObject({ catalogComplete: false, catalogRefreshing: false });

  mocks.refreshModels.mockResolvedValueOnce(undefined);
  mocks.getModels.mockResolvedValueOnce({ models: [], complete: false, refreshing: true });
  await useModelsStore.getState().refreshModels();
  expect(mocks.refreshModels).toHaveBeenCalledOnce();
  expect(useModelsStore.getState()).toMatchObject({ catalogComplete: false, catalogRefreshing: true, error: null });
});
