import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { isCustomParams, resolveModelTools, useModelParamsStore } from "./use-model-params-store";
import { useChatStore } from "./use-chat-store";
import { useModelsStore } from "./use-models-store";
import { ModelKind } from "@/types/models";

const settings = vi.hoisted(() => ({
  getModelParams: vi.fn(),
  getModelChatNamingEnabled: vi.fn(),
  setModelParams: vi.fn(),
}));
const runStream = vi.hoisted(() => vi.fn());
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ settings, chat: { runStream } }) }));

const defaults = useModelParamsStore.getState();
beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  useModelParamsStore.setState(defaults, true);
  useModelsStore.setState({ models: [] });
  settings.getModelChatNamingEnabled.mockResolvedValue(false);
  settings.setModelParams.mockResolvedValue(undefined);
});
afterEach(async () => {
  await vi.runAllTimersAsync();
  vi.useRealTimers();
});

it("loads tool preferences alongside legacy reasoning and sampling settings", async () => {
  settings.getModelParams.mockResolvedValue({
    model: {
      sampling: { type: "Argmax" },
      reasoningEnabled: false,
      modelChatNamingEnabled: true,
      dateTimeToolEnabled: false,
      chartToolEnabled: false,
    },
  });
  await useModelParamsStore.getState().load();
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { type: "Greedy" },
    reasoningEffort: "disabled",
    modelChatNamingEnabled: true,
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
  });
  expect(useModelParamsStore.getState().globalModelChatNamingEnabled).toBe(false);
});

it("ignores malformed tool overrides and inherits current defaults", async () => {
  settings.getModelParams.mockResolvedValue({
    model: {
      sampling: { type: "Default" },
      modelChatNamingEnabled: "false",
      dateTimeToolEnabled: null,
      chartToolEnabled: "false",
    },
  });
  await useModelParamsStore.getState().load();
  expect(resolveModelTools(useModelParamsStore.getState().getParams("model"), false)).toEqual({
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
  });
});

it("persists tool overrides and removes them when parameters are reset", async () => {
  const params = {
    sampling: { type: "Default" as const },
    modelChatNamingEnabled: true,
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
  };
  useModelParamsStore.getState().setParams("model", params);
  await vi.runAllTimersAsync();
  expect(settings.setModelParams).toHaveBeenLastCalledWith("model", params);

  useModelParamsStore.getState().resetParams("model");
  await vi.runAllTimersAsync();
  expect(settings.setModelParams).toHaveBeenLastCalledWith("model", null);
  expect(resolveModelTools(useModelParamsStore.getState().getParams("model"), false)).toEqual({
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
  });
});

it("marks parameters modified only when their effective tools differ from defaults", () => {
  const sampling = { type: "Default" as const };
  expect(isCustomParams({ sampling }, false)).toBe(false);
  expect(isCustomParams({ sampling, modelChatNamingEnabled: false, dateTimeToolEnabled: true }, false)).toBe(false);
  expect(isCustomParams({ sampling, modelChatNamingEnabled: true }, false)).toBe(false);
  expect(isCustomParams({ sampling, modelChatNamingEnabled: false }, true)).toBe(true);
  expect(isCustomParams({ sampling, dateTimeToolEnabled: false }, true)).toBe(true);
  expect(isCustomParams({ sampling, chartToolEnabled: true }, true)).toBe(false);
  expect(isCustomParams({ sampling, chartToolEnabled: false }, true)).toBe(true);
});

it.each([
  { paramSize: 1_999_999_999, expected: false },
  { paramSize: 2_000_000_000, expected: true },
  { paramSize: undefined, expected: true },
])("uses size-specific tool defaults for outgoing requests: %j", ({ paramSize, expected }) => {
  useModelsStore.setState({
    models: [
      {
        repoId: "model",
        name: "Model",
        vendor: "Vendor",
        kind: ModelKind.Text,
        reasoning: { kind: "unsupported" },
        supportsTools: true,
        paramSize,
      },
    ],
  });
  useChatStore.getState().runChatStream({ repoId: "model", messages: [] });
  expect(runStream).toHaveBeenLastCalledWith(
    expect.objectContaining({
      modelChatNamingEnabled: expected,
      dateTimeToolEnabled: expected,
      chartToolEnabled: expected,
    }),
  );
  expect(isCustomParams({ sampling: { type: "Default" } }, true, paramSize)).toBe(false);
});

it("preserves explicit opt-ins on small models while respecting the global naming switch", () => {
  const params = {
    sampling: { type: "Default" as const },
    modelChatNamingEnabled: true,
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
  };
  expect(resolveModelTools(params, true, 1_000_000_000)).toEqual({
    modelChatNamingEnabled: true,
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
  });
  expect(resolveModelTools(params, false, 1_000_000_000)).toEqual({
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
  });
  expect(isCustomParams(params, true, 1_000_000_000)).toBe(true);
  expect(isCustomParams({ sampling: params.sampling, modelChatNamingEnabled: true }, false, 1_000_000_000)).toBe(false);
  expect(isCustomParams({ sampling: params.sampling, dateTimeToolEnabled: false }, true, 1_000_000_000)).toBe(false);
});

it.each([
  { model: undefined, request: undefined, expected: true },
  { model: false, request: undefined, expected: false },
  { model: true, request: false, expected: false },
  { model: false, request: true, expected: true },
])("resolves the model chart preference and explicit run override: %j", (test) => {
  useModelParamsStore.setState({
    paramsByRepoId: { model: { sampling: { type: "Default" }, chartToolEnabled: test.model } },
  });
  useChatStore.getState().runChatStream({ repoId: "model", messages: [], chartToolEnabled: test.request });
  expect(runStream).toHaveBeenLastCalledWith(expect.objectContaining({ chartToolEnabled: test.expected }));
});

it("preserves a model's naming opt-out while the global feature is disabled and re-enabled", () => {
  const params = { sampling: { type: "Default" as const }, modelChatNamingEnabled: false };
  useModelParamsStore.getState().setParams("model", params);

  for (const enabled of [false, true]) {
    useModelParamsStore.getState().setGlobalModelChatNamingEnabled(enabled);
    const saved = useModelParamsStore.getState().getParams("model");
    expect(saved).toEqual(params);
    expect(resolveModelTools(saved, enabled).modelChatNamingEnabled).toBe(false);
  }
});

it.each([
  { global: false, model: true, request: true, expected: false },
  { global: true, model: false, request: true, expected: false },
  { global: true, model: undefined, request: false, expected: false },
  { global: true, model: undefined, request: true, expected: true },
])("enforces the global and per-model naming gates on outgoing requests: %j", (test) => {
  useModelParamsStore.setState({
    globalModelChatNamingEnabled: test.global,
    paramsByRepoId: { model: { sampling: { type: "Default" }, modelChatNamingEnabled: test.model } },
  });
  useChatStore.getState().runChatStream({ repoId: "model", messages: [], modelChatNamingEnabled: test.request });
  expect(runStream).toHaveBeenLastCalledWith(expect.objectContaining({ modelChatNamingEnabled: test.expected }));
});

it("uses model-default reasoning for unset parameters while retaining explicit model overrides", async () => {
  settings.getModelParams.mockResolvedValue({
    legacyOn: { sampling: { type: "Default" }, reasoningEnabled: true },
    legacyOff: { sampling: { type: "Default" }, reasoningEnabled: false },
    explicit: { sampling: { type: "Default" }, reasoningEffort: "high", reasoningEnabled: false },
  });
  await useModelParamsStore.getState().load();
  for (const [repoId, reasoningEffort] of [
    ["unset", "default"],
    ["legacyOn", "default"],
    ["legacyOff", "disabled"],
    ["explicit", "high"],
  ]) {
    useChatStore.getState().runChatStream({ repoId: repoId!, messages: [] });
    expect(runStream).toHaveBeenLastCalledWith(expect.objectContaining({ repoId, reasoningEffort }));
  }
  expect(isCustomParams({ sampling: { type: "Default" }, reasoningEffort: "default" }, true)).toBe(false);
  expect(isCustomParams({ sampling: { type: "Default" }, reasoningEffort: "disabled" }, true)).toBe(false);
});
