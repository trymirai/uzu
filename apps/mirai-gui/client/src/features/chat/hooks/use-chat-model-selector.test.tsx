import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { useChatStore } from "@/stores/use-chat-store";
import { useModelsStore } from "@/stores/use-models-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { modelDownloadPhases } from "@/types/model-manager";
import { ModelKind, type PlatformModel } from "@/types/models";
import { useChatModelSelector } from "./use-chat-model-selector";

const chatDefaults = useChatStore.getState();
const modelsDefaults = useModelsStore.getState();
const runtimeDefaults = useRuntimeSessionStore.getState();
const model: PlatformModel = {
  repoId: "vendor/model",
  name: "Model",
  vendor: "Vendor",
  kind: ModelKind.Text,
  reasoning: { kind: "unsupported" },
  supportsTools: false,
};

beforeEach(() => {
  useChatStore.setState({ ...chatDefaults, lastUsedModel: { modelId: model.repoId, modelName: model.name } }, true);
  useModelsStore.setState({ ...modelsDefaults, initialized: true, hasLoadedModels: true }, true);
  useRuntimeSessionStore.setState(runtimeDefaults, true);
});
afterEach(cleanup);

it("preserves the last used model through partial catalog and storage initialization", () => {
  const { result } = renderHook(() => useChatModelSelector({ chatId: "chat" }));
  expect(result.current.currentModelId).toBe(model.repoId);
  expect(result.current.hasAvailableModels).toBe(false);

  act(() =>
    useModelsStore.setState({
      models: [model],
      catalogComplete: true,
      modelPhasesById: { [model.repoId]: modelDownloadPhases.initializing },
    }),
  );
  expect(result.current.currentModelId).toBe(model.repoId);
  expect(result.current.hasAvailableModels).toBe(false);

  act(() => useModelsStore.setState({ modelPhasesById: { [model.repoId]: modelDownloadPhases.downloaded } }));
  expect(result.current.currentModelId).toBe(model.repoId);
  expect(result.current.hasAvailableModels).toBe(true);
});

it("clears a missing model only once the catalog is complete", () => {
  const { result } = renderHook(() => useChatModelSelector({ chatId: "chat" }));
  expect(result.current.currentModelId).toBe(model.repoId);

  act(() => useModelsStore.setState({ catalogComplete: true }));
  expect(result.current.currentModelId).toBeUndefined();
});

it("clears a selected model after initialization confirms it is not downloaded", () => {
  useModelsStore.setState({ models: [model], modelPhasesById: { [model.repoId]: modelDownloadPhases.initializing } });
  const { result } = renderHook(() => useChatModelSelector({ chatId: "chat" }));
  expect(result.current.currentModelId).toBe(model.repoId);

  act(() =>
    useModelsStore.setState({
      catalogComplete: true,
      modelPhasesById: { [model.repoId]: modelDownloadPhases.notDownloaded },
    }),
  );
  expect(result.current.currentModelId).toBeUndefined();
});
