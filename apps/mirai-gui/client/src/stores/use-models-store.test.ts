import { beforeEach, expect, it, vi } from "vitest";
import type { EngineModel } from "@/types/model-manager";
import { modelDownloadPhases } from "@/types/model-manager";
import { useModelsStore } from "./use-models-store";

const mocks = vi.hoisted(() => ({ getModels: vi.fn() }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ models: { getModels: mocks.getModels } }) }));

const REPO_ID = "vendor/model";
const defaults = useModelsStore.getState();

const snapshot = (phase: EngineModel["state"]["phase"], seq: number): EngineModel => ({
  identifier: REPO_ID,
  repoId: REPO_ID,
  vendor: "Vendor",
  name: "Model",
  reasoning: { kind: "unsupported" },
  state: { phase, totalKbytes: 100, downloadedKbytes: 50, seq },
});

beforeEach(() => {
  vi.clearAllMocks();
  useModelsStore.setState(defaults, true);
});

it("keeps a download event that arrived while the catalog request was in flight", async () => {
  let resolveModels!: (value: EngineModel[]) => void;
  mocks.getModels.mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveModels = resolve;
      }),
  );
  const fetching = useModelsStore.getState().fetchModels();

  useModelsStore.getState().applyDownloadEvent({ kind: "done", identifier: REPO_ID, seq: 7 });
  resolveModels([snapshot(modelDownloadPhases.downloading, 3)]);
  await fetching;

  expect(useModelsStore.getState().modelStatesById[REPO_ID]?.phase).toBe(modelDownloadPhases.downloaded);
});

it("applies a snapshot that is newer than the last event", async () => {
  useModelsStore.getState().applyDownloadEvent({ kind: "paused", identifier: REPO_ID, seq: 2 });
  mocks.getModels.mockResolvedValue([snapshot(modelDownloadPhases.downloaded, 5)]);

  await useModelsStore.getState().fetchModels();

  expect(useModelsStore.getState().modelStatesById[REPO_ID]?.phase).toBe(modelDownloadPhases.downloaded);
});

it("ignores an event older than the state it already holds", () => {
  useModelsStore.getState().applyDownloadEvent({ kind: "done", identifier: REPO_ID, seq: 9 });
  useModelsStore.getState().applyDownloadEvent({ kind: "paused", identifier: REPO_ID, seq: 4 });

  expect(useModelsStore.getState().modelStatesById[REPO_ID]?.phase).toBe(modelDownloadPhases.downloaded);
});
