import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import { ModelKind, type PlatformModel } from "@/types/models";
import { ModelParamsControls } from "./model-params-controls";
import { ModelParamsDrawer } from "./model-params-drawer";
import type { SamplingDefaults } from "@/platform/services/chat";

const mocks = vi.hoisted(() => ({
  setModelParams: vi.fn(async () => {}),
  getSamplingDefaults: vi.fn<() => Promise<SamplingDefaults | null>>(),
}));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ settings: mocks, chat: mocks }) }));
const samplingDefaults = {
  type: "Stochastic",
  temperature: 0.6,
  topK: null,
  topP: null,
  minP: 0.05,
  repetitionPenalty: 1.1,
  suffixRepetitionLength: 64,
} as const;
const paramsDefaults = useModelParamsStore.getState();
const modelsDefaults = useModelsStore.getState();
const runtimeDefaults = useRuntimeSessionStore.getState();
const model: PlatformModel = {
  repoId: "model",
  name: "Model",
  vendor: "Vendor",
  kind: ModelKind.Text,
  reasoning: { kind: "toggle", defaultEffort: "default" },
  supportsTools: true,
  paramSize: 2_000_000_000,
};
const parameterRow = (name: string) => within(screen.getByRole("textbox", { name }).parentElement!.parentElement!);

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  mocks.getSamplingDefaults.mockResolvedValue(samplingDefaults);
  useModelParamsStore.setState(paramsDefaults, true);
  useModelsStore.setState({ ...modelsDefaults, models: [model] }, true);
  useRuntimeSessionStore.setState(runtimeDefaults, true);
});

it("displays the model's resolved defaults without creating a sampling override", async () => {
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  await act(async () => {});
  const options = screen.getAllByRole("radio");
  expect(options.map((option) => option.getAttribute("aria-label"))).toEqual(["Stochastic", "Greedy"]);
  expect(options[0]!.getAttribute("aria-checked")).toBe("true");
  expect(screen.queryByRole("radio", { name: "Default" })).toBeNull();
  expect((screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement).value).toBe("0.6");
  expect((screen.getByRole("textbox", { name: "Top K" }) as HTMLInputElement).disabled).toBe(true);
  expect((screen.getByRole("textbox", { name: "Top P" }) as HTMLInputElement).disabled).toBe(true);
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
  expect(screen.getAllByRole("switch").map((element) => element.getAttribute("aria-label"))).toEqual([
    "Current date and time",
    "Draw charts",
    "Name chat using a tool",
  ]);
  expect(screen.getByRole("switch", { name: "Draw charts" }).getAttribute("aria-checked")).toBe("true");
  expect(useModelParamsStore.getState().paramsByRepoId).toEqual({});
});

it("preserves disabled filters and repetition settings when editing a resolved default", async () => {
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  await act(async () => {});
  fireEvent.change(screen.getByRole("textbox", { name: "Temperature" }), { target: { value: "0.8" } });
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ ...samplingDefaults, temperature: 0.8 });
  expect(parameterRow("Temperature").getByTitle("Changed from default").getAttribute("aria-hidden")).toBe("true");
  expect(screen.getByRole("button", { name: "Reset sampling to defaults" })).toBeTruthy();
  expect(within(screen.getByRole("radio", { name: "Stochastic" })).queryByTitle("Changed from default")).toBeNull();
  fireEvent.change(screen.getByRole("textbox", { name: "Temperature" }), { target: { value: "0.6" } });
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Default" });
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();

  fireEvent.click(screen.getByRole("checkbox", { name: "Enable Top K" }));
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ ...samplingDefaults, topK: 40 });
  expect(parameterRow("Top K").getByTitle("Changed from default")).toBeTruthy();
  fireEvent.click(screen.getByRole("checkbox", { name: "Enable Top K" }));
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Default" });
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
});

it("marks changes to enabled parameters and disabled filters until sampling is reset", async () => {
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  await act(async () => {});

  fireEvent.click(screen.getByRole("checkbox", { name: "Enable Top P" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Enable Min P" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Repetition penalty" }), { target: { value: "1.2" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Suffix repetition length" }), { target: { value: "96" } });
  for (const label of ["Top P", "Min P", "Repetition penalty", "Suffix repetition length"]) {
    expect(parameterRow(label).getByTitle("Changed from default")).toBeTruthy();
  }
  expect(screen.getAllByTitle("Changed from default")).toHaveLength(4);
  expect((screen.getByRole("textbox", { name: "Min P" }) as HTMLInputElement).disabled).toBe(true);

  fireEvent.click(screen.getByRole("button", { name: "Reset sampling to defaults" }));
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Default" });
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  expect((screen.getByRole("textbox", { name: "Min P" }) as HTMLInputElement).disabled).toBe(false);
  expect((screen.getByRole("textbox", { name: "Suffix repetition length" }) as HTMLInputElement).value).toBe("64");
});

it("does not mark explicit matching defaults or distinguish null from omitted disabled filters", async () => {
  useModelParamsStore.setState({
    paramsByRepoId: {
      model: {
        sampling: { ...samplingDefaults, temperature: 1, topK: undefined, topP: undefined },
        modelChatNamingEnabled: true,
        dateTimeToolEnabled: true,
        chartToolEnabled: true,
      },
    },
  });
  render(<ModelParamsControls repoId="model" samplingDefaults={{ ...samplingDefaults, temperature: null }} />);
  await act(async () => {});
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
});

it("waits for defaults before switching from greedy to stochastic", async () => {
  let resolveDefaults!: (defaults: SamplingDefaults) => void;
  mocks.getSamplingDefaults.mockReturnValue(new Promise((resolve) => (resolveDefaults = resolve)));
  useModelParamsStore.setState({ paramsByRepoId: { model: { sampling: { type: "Greedy" } } } });
  render(<ModelParamsDrawer open chatId="chat" repoId="model" onClose={() => {}} />);
  expect(screen.getByText("Loading model sampling settings…")).toBeTruthy();
  expect(screen.queryByRole("radio", { name: "Stochastic" })).toBeNull();
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Greedy" });

  await act(async () => resolveDefaults(samplingDefaults));
  const stochastic = screen.getByRole("radio", { name: "Stochastic" }) as HTMLButtonElement;
  expect(stochastic.disabled).toBe(false);
  expect(within(screen.getByRole("radio", { name: "Greedy" })).getByTitle("Changed from default")).toBeTruthy();
  fireEvent.click(stochastic);
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Default" });
  expect((screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement).value).toBe("0.6");
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
});

it.each([null, 0.01, 1.5])(
  "shows the actual model temperature %s without clamping or inventing greedy sampling",
  async (temperature) => {
    render(<ModelParamsControls repoId="model" samplingDefaults={{ ...samplingDefaults, temperature }} />);
    await act(async () => {});
    const input = screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement;
    expect(input.value).toBe(String(temperature ?? 1));
    expect(screen.getByRole("radio", { name: "Stochastic" }).getAttribute("aria-checked")).toBe("true");
    fireEvent.blur(input);
    expect(useModelParamsStore.getState().paramsByRepoId).toEqual({});
  },
);

it("shows no invented default while metadata is unavailable", async () => {
  render(<ModelParamsControls repoId="model" samplingDefaults={null} />);
  await act(async () => {});
  expect(screen.getByText("Model sampling settings are unavailable.")).toBeTruthy();
  expect(screen.getAllByRole("radio").every((option) => option.getAttribute("aria-checked") === "false")).toBe(true);
  expect(screen.queryByRole("textbox", { name: "Temperature" })).toBeNull();
  fireEvent.click(screen.getByRole("radio", { name: "Greedy" }));
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Greedy" });
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Reset sampling to defaults" }));
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Default" });
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
});

it("allows resetting explicit parameters with unknown defaults without inventing per-value markers", async () => {
  useModelParamsStore.setState({
    paramsByRepoId: { model: { sampling: { type: "Stochastic", temperature: 0.8, topK: 20 } } },
  });
  render(<ModelParamsControls repoId="model" samplingDefaults={null} />);
  await act(async () => {});
  expect((screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement).value).toBe("0.8");
  expect(screen.queryByTitle("Changed from default")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Reset sampling to defaults" }));
  expect(useModelParamsStore.getState().getParams("model").sampling).toEqual({ type: "Default" });
  expect(screen.getByText("Model sampling settings are unavailable.")).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
});
afterEach(async () => {
  cleanup();
  await vi.runAllTimersAsync();
  vi.useRealTimers();
});

it("hides chat naming while globally disabled and restores a model's opt-out when re-enabled", async () => {
  useModelParamsStore.setState({ globalModelChatNamingEnabled: false });
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  await act(async () => {});
  expect(screen.queryByRole("switch", { name: "Name chat using a tool" })).toBeNull();
  expect(screen.queryByText("Name chat")).toBeNull();
  expect(screen.getByRole("switch", { name: "Current date and time" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();

  act(() => useModelParamsStore.getState().setGlobalModelChatNamingEnabled(true));
  const naming = screen.getByRole("switch", { name: "Name chat using a tool" });
  expect(naming.getAttribute("aria-checked")).toBe("true");
  fireEvent.click(naming);
  expect(useModelParamsStore.getState().getParams("model").modelChatNamingEnabled).toBe(false);
  expect(within(screen.getByText("Name chat")).getByTitle("Changed from default")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Reset tools to defaults" })).toBeTruthy();

  act(() => useModelParamsStore.getState().setGlobalModelChatNamingEnabled(false));
  expect(screen.queryByRole("switch", { name: "Name chat using a tool" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
  expect(useModelParamsStore.getState().getParams("model").modelChatNamingEnabled).toBe(false);

  act(() => useModelParamsStore.getState().setGlobalModelChatNamingEnabled(true));
  expect(screen.getByRole("switch", { name: "Name chat using a tool" }).getAttribute("aria-checked")).toBe("false");
  expect(within(screen.getByText("Name chat")).getByTitle("Changed from default")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Reset tools to defaults" }));
  expect(useModelParamsStore.getState().getParams("model").modelChatNamingEnabled).toBeUndefined();
  expect(screen.getByRole("switch", { name: "Name chat using a tool" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
});

it("resetting visible tools leaves a hidden naming opt-out unchanged", async () => {
  useModelParamsStore.setState({
    globalModelChatNamingEnabled: false,
    paramsByRepoId: {
      model: {
        sampling: { type: "Default" },
        modelChatNamingEnabled: false,
        dateTimeToolEnabled: false,
        chartToolEnabled: false,
      },
    },
  });
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  await act(async () => {});
  expect(screen.queryByText("Name chat")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Reset tools to defaults" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { type: "Default" },
    modelChatNamingEnabled: false,
  });
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
});

it("marks tool overrides until each switch returns to its default", async () => {
  useModelParamsStore.setState({
    paramsByRepoId: { model: { sampling: { type: "Default" }, reasoningEffort: "disabled" } },
  });
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  await act(async () => {});
  expect(screen.queryByRole("switch", { name: "Reasoning" })).toBeNull();
  fireEvent.click(screen.getByRole("switch", { name: "Current date and time" }));
  fireEvent.click(screen.getByRole("switch", { name: "Name chat using a tool" }));
  fireEvent.click(screen.getByRole("radio", { name: "Greedy" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { type: "Greedy" },
    reasoningEffort: "disabled",
    modelChatNamingEnabled: false,
    dateTimeToolEnabled: false,
  });
  expect(within(screen.getByText("Current date and time")).getByTitle("Changed from default")).toBeTruthy();
  expect(within(screen.getByText("Name chat")).getByTitle("Changed from default")).toBeTruthy();
  fireEvent.click(screen.getByRole("switch", { name: "Current date and time" }));
  expect(within(screen.getByText("Current date and time")).queryByTitle("Changed from default")).toBeNull();
  expect(screen.getByRole("button", { name: "Reset tools to defaults" })).toBeTruthy();
  fireEvent.click(screen.getByRole("switch", { name: "Name chat using a tool" }));
  expect(within(screen.getByText("Name chat")).queryByTitle("Changed from default")).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { type: "Greedy" },
    reasoningEffort: "disabled",
  });
  expect(screen.getByRole("switch", { name: "Name chat using a tool" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.getByRole("switch", { name: "Current date and time" }).getAttribute("aria-checked")).toBe("true");
});

it("disables tools for a model without tool support", () => {
  useModelsStore.setState({ models: [{ ...model, supportsTools: false }] });
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  expect(screen.getByText("This model does not support tool calls.")).toBeTruthy();
  expect((screen.getByRole("switch", { name: "Name chat using a tool" }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("switch", { name: "Current date and time" }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("switch", { name: "Draw charts" }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.getByRole("switch", { name: "Draw charts" }).getAttribute("aria-checked")).toBe("false");
});

it("defaults all tools off below 2B while allowing per-model opt-ins and reset", () => {
  useModelsStore.setState({ models: [{ ...model, paramSize: 1_999_999_999 }] });
  render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
  const labels = ["Current date and time", "Draw charts", "Name chat using a tool"];
  for (const name of labels) {
    const toggle = screen.getByRole("switch", { name }) as HTMLButtonElement;
    expect(toggle.disabled).toBe(false);
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-checked")).toBe("true");
  }
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { type: "Default" },
    dateTimeToolEnabled: true,
    chartToolEnabled: true,
    modelChatNamingEnabled: true,
  });
  fireEvent.click(screen.getByRole("button", { name: "Reset tools to defaults" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({ sampling: { type: "Default" } });
  for (const name of labels) expect(screen.getByRole("switch", { name }).getAttribute("aria-checked")).toBe("false");
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
});

it("keeps chart opt-outs per model and resets them to the inherited enabled default", () => {
  useModelsStore.setState({ models: [model, { ...model, repoId: "other" }] });
  const { rerender } = render(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  fireEvent.click(screen.getByRole("switch", { name: "Draw charts" }));
  expect(useModelParamsStore.getState().getParams("model").chartToolEnabled).toBe(false);
  expect(within(screen.getByText("Draw charts")).getByTitle("Changed from default")).toBeTruthy();

  rerender(<ModelParamsControls repoId="other" samplingDefaults={samplingDefaults} />);
  expect(screen.getByRole("switch", { name: "Draw charts" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();

  rerender(<ModelParamsControls repoId="model" samplingDefaults={samplingDefaults} />);
  expect(screen.getByRole("switch", { name: "Draw charts" }).getAttribute("aria-checked")).toBe("false");
  fireEvent.click(screen.getByRole("button", { name: "Reset tools to defaults" }));
  expect(useModelParamsStore.getState().getParams("model").chartToolEnabled).toBeUndefined();
  expect(screen.getByRole("switch", { name: "Draw charts" }).getAttribute("aria-checked")).toBe("true");
  expect(within(screen.getByText("Draw charts")).queryByTitle("Changed from default")).toBeNull();
});

it("prefetches defaults while closed and opens ready controls without refetching or expanding them", async () => {
  const { rerender } = render(<ModelParamsDrawer open={false} chatId="chat" repoId="model" onClose={() => {}} />);
  await act(async () => {});
  expect(mocks.getSamplingDefaults).toHaveBeenCalledExactlyOnceWith("model");
  expect(screen.queryByRole("textbox", { name: "Temperature" })).toBeNull();

  rerender(<ModelParamsDrawer open chatId="chat" repoId="model" onClose={() => {}} />);
  const temperature = screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement;
  expect(temperature.value).toBe("0.6");
  const panel = temperature.closest<HTMLElement>("[style]")!;
  expect(panel.style.height).toBe("auto");
  expect(panel.style.opacity).toBe("1");
  expect(screen.queryByText("Loading model sampling settings…")).toBeNull();
  expect(mocks.getSamplingDefaults).toHaveBeenCalledTimes(1);
  expect(useModelParamsStore.getState().paramsByRepoId).toEqual({});
});

it("mounts controls at their final height when defaults arrive after the drawer opens", async () => {
  let resolveDefaults!: (defaults: SamplingDefaults) => void;
  mocks.getSamplingDefaults.mockReturnValue(new Promise((resolve) => (resolveDefaults = resolve)));
  render(<ModelParamsDrawer open chatId="chat" repoId="model" onClose={() => {}} />);
  expect(screen.getByText("Loading model sampling settings…")).toBeTruthy();
  expect(screen.queryByRole("textbox", { name: "Temperature" })).toBeNull();

  await act(async () => resolveDefaults(samplingDefaults));
  const temperature = screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement;
  expect(temperature.value).toBe("0.6");
  const panel = temperature.closest<HTMLElement>("[style]")!;
  expect(panel.style.height).toBe("auto");
  expect(panel.style.opacity).toBe("1");
  expect(screen.queryByText("Loading model sampling settings…")).toBeNull();
});

it("retries unavailable prefetched defaults on opening and mounts recovered controls without expanding", async () => {
  let resolveRetry!: (defaults: SamplingDefaults) => void;
  mocks.getSamplingDefaults
    .mockResolvedValueOnce(null)
    .mockReturnValueOnce(new Promise((resolve) => (resolveRetry = resolve)));
  const { rerender } = render(<ModelParamsDrawer open={false} chatId="chat" repoId="model" onClose={() => {}} />);
  await act(async () => {});
  expect(mocks.getSamplingDefaults).toHaveBeenCalledTimes(1);

  rerender(<ModelParamsDrawer open chatId="chat" repoId="model" onClose={() => {}} />);
  expect(mocks.getSamplingDefaults).toHaveBeenCalledTimes(2);
  await act(async () => resolveRetry(samplingDefaults));
  const temperature = screen.getByRole("textbox", { name: "Temperature" }) as HTMLInputElement;
  expect(temperature.value).toBe("0.6");
  const panel = temperature.closest<HTMLElement>("[style]")!;
  expect(panel.style.height).toBe("auto");
  expect(panel.style.opacity).toBe("1");
  expect(mocks.getSamplingDefaults).toHaveBeenCalledTimes(2);
});

it("ignores stale defaults when the selected model changes during a request", async () => {
  let resolveOldDefaults!: (defaults: SamplingDefaults) => void;
  let resolveNewDefaults!: (defaults: SamplingDefaults) => void;
  mocks.getSamplingDefaults
    .mockReturnValueOnce(new Promise((resolve) => (resolveOldDefaults = resolve)))
    .mockReturnValueOnce(new Promise((resolve) => (resolveNewDefaults = resolve)));
  const { rerender } = render(<ModelParamsDrawer open chatId="chat" repoId="model" onClose={() => {}} />);
  rerender(<ModelParamsDrawer open chatId="chat" repoId="other-model" onClose={() => {}} />);
  await act(async () => resolveOldDefaults(samplingDefaults));
  expect(screen.getByText("Loading model sampling settings…")).toBeTruthy();
  expect(screen.queryByRole("textbox", { name: "Temperature" })).toBeNull();

  await act(async () => resolveNewDefaults({ type: "Greedy" }));
  expect(screen.getByRole("radio", { name: "Greedy" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.queryByRole("textbox", { name: "Temperature" })).toBeNull();
  expect(mocks.getSamplingDefaults.mock.calls).toEqual([["model"], ["other-model"]]);
});

it("resets sampling and tools independently while preserving the composer reasoning choice", async () => {
  useModelParamsStore.setState({
    paramsByRepoId: {
      model: {
        sampling: { type: "Greedy" },
        reasoningEffort: "disabled",
        dateTimeToolEnabled: false,
        chartToolEnabled: false,
        modelChatNamingEnabled: false,
      },
    },
  });
  render(<ModelParamsDrawer open chatId="chat" repoId="model" onClose={() => {}} />);
  await act(async () => {});
  expect(screen.queryByRole("button", { name: "Reset to defaults" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Reset sampling to defaults" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { type: "Default" },
    reasoningEffort: "disabled",
    dateTimeToolEnabled: false,
    chartToolEnabled: false,
    modelChatNamingEnabled: false,
  });
  expect(screen.queryByRole("button", { name: "Reset sampling to defaults" })).toBeNull();
  expect(screen.getByRole("button", { name: "Reset tools to defaults" })).toBeTruthy();

  fireEvent.change(screen.getByRole("textbox", { name: "Temperature" }), { target: { value: "0.8" } });
  fireEvent.click(screen.getByRole("button", { name: "Reset tools to defaults" }));
  expect(useModelParamsStore.getState().getParams("model")).toEqual({
    sampling: { ...samplingDefaults, temperature: 0.8 },
    reasoningEffort: "disabled",
  });
  expect(screen.queryByRole("button", { name: "Reset tools to defaults" })).toBeNull();
  expect(screen.getByRole("button", { name: "Reset sampling to defaults" })).toBeTruthy();
  expect(screen.getByRole("switch", { name: "Name chat using a tool" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.getByRole("switch", { name: "Current date and time" }).getAttribute("aria-checked")).toBe("true");
  expect(screen.getByRole("switch", { name: "Draw charts" }).getAttribute("aria-checked")).toBe("true");
});
