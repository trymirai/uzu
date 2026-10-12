import { Channel } from "@tauri-apps/api/core";
import { invoke } from "../shared/invoke";
import type { LlmRunParams } from "@/types/llm-stream";
import type { ChatService, SamplingDefaults, TitleGenParams } from ".";
import { runLlmStream, type RunEvent, type RunTransport } from "./run-stream";

const cancelRun = async (runId: string): Promise<void> => {
  await invoke("cancel_run", { runId });
};

const tauriRunTransport: RunTransport = {
  start: async (runId, params, onEvent) => {
    const channel = new Channel<RunEvent>();
    channel.onmessage = onEvent;
    await invoke("run_stream", { payload: { runId, ...params }, onEvent: channel });
  },
  cancel: cancelRun,
};

export const tauriChat: ChatService = {
  runStream: (params: LlmRunParams) => runLlmStream(tauriRunTransport, params),
  cancelRun,

  generateTitle: (params: TitleGenParams) => invoke<string>("title_gen", { payload: params }),

  cancelTitleGen: async () => {
    await invoke("cancel_title_gen");
  },

  getSamplingDefaults: async (repoId: string) => {
    try {
      return await invoke<SamplingDefaults | null>("chat_sampling_defaults", { repoId });
    } catch (e) {
      console.warn("chat_sampling_defaults failed", { repoId, error: String(e) });
      return null;
    }
  },
};
