import type { LlmRunParams } from "@/types/llm-stream";
import type { TitleGenParams } from ".";
import type { BrowserEncodingSupport } from "./browser-protocol";
import type { RunEvent } from "./run-stream";
import { normalizeChatName, titlePrompt } from "./browser-tools";

type TitleSession = {
  run(params: LlmRunParams & { maxOutputTokens: number }, onEvent: (event: RunEvent) => void): Promise<void>;
};

export async function generateBrowserTitle(
  session: TitleSession,
  params: TitleGenParams,
  support: BrowserEncodingSupport | undefined,
  isCancelled: () => boolean,
): Promise<string> {
  // As on desktop, a checkpoint that always reasons can exhaust the title's
  // small budget without producing an answer. Use the checkpoint's capabilities.
  if (support?.reasoning.kind === "alwaysOn") return "";
  const reasoningEffort =
    support?.reasoning.kind === "toggle"
      ? "disabled"
      : support?.reasoning.kind === "levels"
        ? support.reasoning.efforts[0]
        : "default";
  let title = "";
  await session.run(
    {
      repoId: params.repoId,
      messages: [{ role: "user", content: titlePrompt(params.userText) }],
      reasoningEffort,
      samplingPolicy: { type: "Default" },
      maxOutputTokens: 40,
    },
    (event) => {
      if (event.type === "done" && event.finishReason !== "Cancelled") title = event.text;
    },
  );
  return isCancelled() || !title.trim() ? "" : normalizeChatName(title);
}
