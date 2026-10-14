import { expect, it, vi } from "vitest";
import { restrictCheckpointCapabilities } from "../models/browser-catalog";
import type { BrowserEncodingSupport } from "./browser-protocol";
import { generateBrowserTitle } from "./browser-title";
import { browserTools } from "./browser-tools";
import { emptyStats } from "./empty-stats";

const params = { repoId: "liquid/lfm2.5-230m", userText: "Explain rainbows" };
const encoding: BrowserEncodingSupport = { reasoning: { kind: "alwaysOn" }, supportsTools: true };
type TitleSession = Parameters<typeof generateBrowserTitle>[0];
const titleSession = () => ({
  run: vi.fn<TitleSession["run"]>(async (_params, onEvent) => {
    onEvent({ type: "done", text: '"Rainbow Colors"', stats: emptyStats(), finishReason: "Stop" });
  }),
});

it("generates a bounded title for a non-reasoning checkpoint using a reasoning-capable encoding", async () => {
  const support = restrictCheckpointCapabilities(encoding, { reasoning: false, tools: false });
  const session = titleSession();
  expect(await generateBrowserTitle(session, params, support, () => false)).toBe("Rainbow Colors");
  expect(session.run).toHaveBeenCalledOnce();
  expect(session.run.mock.calls[0]?.[0]).toMatchObject({
    repoId: params.repoId,
    reasoningEffort: "default",
    maxOutputTokens: 40,
  });
  expect(
    browserTools(
      { repoId: params.repoId, messages: [], modelChatNamingEnabled: true, dateTimeToolEnabled: true },
      support.supportsTools,
    ).params.tools,
  ).toEqual([]);
});
it("keeps the desktop title skip for genuinely always-on reasoning checkpoints", async () => {
  const session = titleSession();
  expect(await generateBrowserTitle(session, params, encoding, () => false)).toBe("");
  expect(session.run).not.toHaveBeenCalled();
});
it("uses the cheapest reasoning mode for titles and discards cancelled results", async () => {
  const session = titleSession();
  await generateBrowserTitle(
    session,
    params,
    { ...encoding, reasoning: { kind: "toggle", defaultEffort: "default" } },
    () => false,
  );
  expect(session.run.mock.calls[0]?.[0].reasoningEffort).toBe("disabled");
  expect(
    await generateBrowserTitle(session, params, { ...encoding, reasoning: { kind: "unsupported" } }, () => true),
  ).toBe("");
});
