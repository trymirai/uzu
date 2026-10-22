import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { ModelKind, type PlatformModel } from "@/types/models";
import type { ReasoningSupport } from "@/types/sampling";
import { ChatInput } from ".";
import { ReasoningSelector } from "./reasoning-selector";

const settings = vi.hoisted(() => ({ setModelParams: vi.fn(async () => {}) }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ settings }) }));

const paramsDefaults = useModelParamsStore.getState();
const modelsDefaults = useModelsStore.getState();
const model: PlatformModel = {
  repoId: "model",
  name: "Model",
  vendor: "Vendor",
  kind: ModelKind.Text,
  reasoning: { kind: "toggle", defaultEffort: "default" },
  supportsTools: true,
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  useModelParamsStore.setState(paramsDefaults, true);
  useModelsStore.setState({ ...modelsDefaults, models: [model] }, true);
});

afterEach(async () => {
  cleanup();
  await vi.runAllTimersAsync();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it.each<ReasoningSupport>([
  { kind: "unsupported" },
  { kind: "alwaysOn" },
  { kind: "levels", efforts: [] },
  { kind: "levels", efforts: ["high"], defaultEffort: "high" },
])("hides a selector with no meaningful choice: %j", (reasoning) => {
  useModelsStore.setState({ models: [{ ...model, reasoning }] });
  render(<ReasoningSelector repoId="model" />);
  expect(screen.queryByRole("button")).toBeNull();
});

it("sits between the model picker and settings and follows the picker busy state", () => {
  const props = {
    value: "",
    onChange: vi.fn(),
    models: [{ id: model.repoId, name: model.name, logo: null }],
    activeModelId: model.repoId,
    onModelSettingsClick: vi.fn(),
  };
  const { rerender } = render(<ChatInput {...props} />);
  const modelButton = screen.getByRole("button", { name: model.name });
  const reasoningButton = screen.getByRole("button", { name: /^Reasoning:/ });
  const settingsButton = screen.getByRole("button", { name: "Model settings" });
  expect(modelButton.compareDocumentPosition(reasoningButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(reasoningButton.compareDocumentPosition(settingsButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

  rerender(<ChatInput {...props} modelPickerDisabled />);
  expect((reasoningButton as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(reasoningButton);
  expect(screen.queryByRole("menu")).toBeNull();
});

it("cannot change reasoning if generation starts while the menu is already open", async () => {
  const { rerender } = render(<ReasoningSelector repoId="model" />);
  fireEvent.click(screen.getByRole("button", { name: /^Reasoning:/ }));
  await screen.findByRole("menu");
  rerender(<ReasoningSelector repoId="model" disabled />);
  const off = screen.getByRole("menuitem", { name: "Off" });
  expect(off.getAttribute("aria-disabled")).toBe("true");
  fireEvent.click(off);
  expect(useModelParamsStore.getState().getParams("model").reasoningEffort).toBeUndefined();
});

it("shows actual levels, marks the model default, and clears only its reasoning override when selected", async () => {
  const otherParams = {
    sampling: { type: "Greedy" as const },
    dateTimeToolEnabled: false,
    modelChatNamingEnabled: false,
  };
  useModelsStore.setState({
    models: [
      { ...model, reasoning: { kind: "levels", efforts: ["low", "medium", "high", "high"], defaultEffort: "medium" } },
    ],
  });
  useModelParamsStore.setState({ paramsByRepoId: { model: { ...otherParams, reasoningEffort: "high" } } });
  render(<ReasoningSelector repoId="model" />);
  const trigger = screen.getByRole("button", { name: "Reasoning: High" });
  expect(useModelParamsStore.getState().getParams("model").reasoningEffort).toBe("high");
  expect(settings.setModelParams).not.toHaveBeenCalled();

  fireEvent.click(trigger);
  await screen.findByRole("menu");
  expect(screen.getAllByRole("menuitem").map((item) => item.getAttribute("aria-label"))).toEqual([
    "High",
    "Medium, model default",
    "Low",
  ]);
  expect(screen.queryByRole("menuitem", { name: "Default" })).toBeNull();
  expect(screen.queryByRole("menuitem", { name: "Off" })).toBeNull();
  expect(screen.getByTitle("Model default").textContent).toBe("default");
  fireEvent.click(screen.getByRole("menuitem", { name: "Medium, model default" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual(otherParams);
  expect(screen.getByRole("button", { name: "Reasoning: Medium, model default" })).toBeTruthy();

  fireEvent.click(trigger);
  await screen.findByRole("menu");
  fireEvent.click(screen.getByRole("menuitem", { name: "Low" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({ ...otherParams, reasoningEffort: "low" });
});

it("does not invent a selected level or default marker when the model default is unknown", async () => {
  useModelsStore.setState({ models: [{ ...model, reasoning: { kind: "levels", efforts: ["low", "high"] } }] });
  render(<ReasoningSelector repoId="model" />);
  fireEvent.click(screen.getByRole("button", { name: "Reasoning" }));
  await screen.findByRole("menu");
  expect(screen.getAllByRole("menuitem").every((item) => !item.hasAttribute("aria-current"))).toBe(true);
  expect(screen.queryByTitle("Model default")).toBeNull();
  fireEvent.click(screen.getByRole("menuitem", { name: "High" }));
  expect(screen.getByRole("button", { name: "Reasoning: High" })).toBeTruthy();
  expect(useModelParamsStore.getState().getParams("model").reasoningEffort).toBe("high");
});

it("switches choices and stored selection with the active model without changing another model's parameters", async () => {
  useModelsStore.setState({
    models: [
      model,
      { ...model, repoId: "levels", reasoning: { kind: "levels", efforts: ["low", "high"], defaultEffort: "low" } },
    ],
  });
  useModelParamsStore.setState({
    paramsByRepoId: {
      model: { sampling: { type: "Greedy" }, reasoningEffort: "disabled" },
      levels: { sampling: { type: "Default" }, reasoningEffort: "high" },
    },
  });
  const { rerender } = render(<ReasoningSelector repoId="levels" />);
  expect(screen.getByRole("button", { name: "Reasoning: High" })).toBeTruthy();
  rerender(<ReasoningSelector repoId="model" />);
  fireEvent.click(screen.getByRole("button", { name: "Reasoning: Off" }));
  await screen.findByRole("menu");
  expect(screen.getAllByRole("menuitem").map((item) => item.getAttribute("aria-label"))).toEqual([
    "Thinking, model default",
    "Off",
  ]);
  fireEvent.click(screen.getByRole("menuitem", { name: "Thinking, model default" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({ sampling: { type: "Greedy" } });
  expect(useModelParamsStore.getState().getParams("levels").reasoningEffort).toBe("high");
});

it("keeps keyboard navigation and activation working through the selection rows", async () => {
  render(<ReasoningSelector repoId="model" />);
  const trigger = screen.getByRole("button", { name: "Reasoning: Thinking, model default" });
  act(() => trigger.focus());
  fireEvent.keyDown(trigger, { key: "ArrowDown" });
  const menu = await screen.findByRole("menu");
  fireEvent.keyDown(menu, { key: "End" });
  const off = screen.getByRole("menuitem", { name: "Off" });
  await waitFor(() => expect(menu.getAttribute("aria-activedescendant")).toBe(off.id));
  fireEvent.keyDown(menu, { key: "Enter" });
  expect(useModelParamsStore.getState().getParams("model").reasoningEffort).toBe("disabled");
  expect(screen.getByRole("button", { name: "Reasoning: Off" })).toBeTruthy();
});

it("orders all supported reasoning levels from highest to lowest with Off last", async () => {
  useModelsStore.setState({
    models: [
      {
        ...model,
        reasoning: {
          kind: "levels",
          efforts: ["disabled", "medium", "xhigh", "low", "high", "medium"],
          defaultEffort: "medium",
        },
      },
    ],
  });
  render(<ReasoningSelector repoId="model" />);
  fireEvent.click(screen.getByRole("button", { name: "Reasoning: Medium, model default" }));
  await screen.findByRole("menu");
  expect(screen.getAllByRole("menuitem").map((item) => item.getAttribute("aria-label"))).toEqual([
    "XHigh",
    "High",
    "Medium, model default",
    "Low",
    "Off",
  ]);
});
