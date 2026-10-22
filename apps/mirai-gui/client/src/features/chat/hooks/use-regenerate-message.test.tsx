import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useChatStore } from "@/stores/use-chat-store";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useModelParamsStore } from "@/stores/use-model-params-store";
import { useModelsStore } from "@/stores/use-models-store";
import { ModelKind } from "@/types/models";
import { useRuntimeSessionStore } from "@/stores/use-runtime-session-store";
import type { ToastApi } from "@/components/ui/toast/use-toast";
import { Roles } from "@/types/chat";
import { useRegenerateMessage } from "./use-regenerate-message";

const settings = vi.hoisted(() => ({ getModelChatNamingEnabled: vi.fn(async () => true) }));
vi.mock("@/platform/platform-singleton", () => ({ getPlatform: () => ({ settings }) }));
const chatDefaults = useChatStore.getState();
const sessionDefaults = useChatSessionStore.getState();
const paramsDefaults = useModelParamsStore.getState();
const runtimeDefaults = useRuntimeSessionStore.getState();
const startStream = vi.fn(async () => {});
const toast = { error: vi.fn(), info: vi.fn(), warning: vi.fn() } as unknown as ToastApi;

beforeEach(() => {
  vi.clearAllMocks();
  useChatStore.setState(
    {
      ...chatDefaults,
      currentChatId: "chat",
      messages: [
        { id: "user", sender: Roles.User, text: "Question", timestamp: 1 },
        {
          id: "answer",
          sender: Roles.Assistant,
          text: "Answer",
          timestamp: 2,
          modelId: "original",
          modelName: "Original",
        },
      ],
      persistMessagePatch: vi.fn(async () => {}),
    },
    true,
  );
  useChatSessionStore.setState(sessionDefaults, true);
  useModelParamsStore.setState(paramsDefaults, true);
  useModelsStore.setState({ models: [] });
  useRuntimeSessionStore.setState(runtimeDefaults, true);
});
afterEach(cleanup);

it("regenerates with the selected model's tool overrides", async () => {
  useModelParamsStore.setState({
    paramsByRepoId: {
      original: { sampling: { type: "Default" }, modelChatNamingEnabled: true },
      replacement: {
        sampling: { type: "Default" },
        modelChatNamingEnabled: false,
        dateTimeToolEnabled: false,
        chartToolEnabled: false,
      },
    },
  });
  const hook = renderHook(() => useRegenerateMessage({ chatId: "chat", globalInstructions: "", toast, startStream }));
  await act(() => hook.result.current("answer", "replacement", "Replacement"));
  expect(startStream).toHaveBeenCalledWith(
    expect.objectContaining({
      repoId: "replacement",
      signal: expect.any(AbortSignal),
      modelChatNamingEnabled: false,
      dateTimeToolEnabled: false,
      chartToolEnabled: false,
    }),
  );
});

it.each(["save", "settings"] as const)("does not start regeneration canceled during %s preparation", async (stage) => {
  let resume!: () => void;
  const pending = new Promise<void>((resolve) => {
    resume = resolve;
  });
  const persist = vi.mocked(useChatStore.getState().persistMessagePatch);
  if (stage === "save") persist.mockReturnValueOnce(pending);
  else
    settings.getModelChatNamingEnabled.mockImplementationOnce(async () => {
      await pending;
      return true;
    });
  const hook = renderHook(() => useRegenerateMessage({ chatId: "chat", globalInstructions: "", toast, startStream }));
  let operation!: Promise<void>;
  act(() => {
    operation = hook.result.current("answer", "replacement", "Replacement");
  });
  await waitFor(() => expect(stage === "save" ? persist : settings.getModelChatNamingEnabled).toHaveBeenCalledOnce());
  await act(() => useChatSessionStore.getState().cancelActiveRunForChat("chat"));
  await act(async () => {
    resume();
    await operation;
  });

  expect(startStream).not.toHaveBeenCalled();
  expect(useChatSessionStore.getState()).toMatchObject({ loadingMessage: null, operationState: "idle" });
});

it("regenerates with current global naming and default tool settings when there is no override", async () => {
  settings.getModelChatNamingEnabled.mockResolvedValueOnce(false);
  const hook = renderHook(() => useRegenerateMessage({ chatId: "chat", globalInstructions: "", toast, startStream }));
  await act(() => hook.result.current("answer", "replacement", "Replacement"));
  expect(startStream).toHaveBeenCalledWith(
    expect.objectContaining({ modelChatNamingEnabled: false, dateTimeToolEnabled: true, chartToolEnabled: true }),
  );
});

it.each([false, true])("regenerates on a small model with tools enabled only after opt-in=%s", async (enabled) => {
  useModelsStore.setState({
    models: [
      {
        repoId: "replacement",
        name: "Replacement",
        vendor: "Vendor",
        kind: ModelKind.Text,
        reasoning: { kind: "unsupported" },
        supportsTools: true,
        paramSize: 1_000_000_000,
      },
    ],
  });
  if (enabled)
    useModelParamsStore.setState({
      paramsByRepoId: {
        replacement: {
          sampling: { type: "Default" },
          modelChatNamingEnabled: true,
          dateTimeToolEnabled: true,
          chartToolEnabled: true,
        },
      },
    });
  const hook = renderHook(() => useRegenerateMessage({ chatId: "chat", globalInstructions: "", toast, startStream }));
  await act(() => hook.result.current("answer", "replacement", "Replacement"));
  expect(startStream).toHaveBeenCalledWith(
    expect.objectContaining({
      modelChatNamingEnabled: enabled,
      dateTimeToolEnabled: enabled,
      chartToolEnabled: enabled,
    }),
  );
});
