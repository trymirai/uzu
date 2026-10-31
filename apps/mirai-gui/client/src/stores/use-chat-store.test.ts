import { beforeEach, expect, it } from "vitest";
import { useChatStore } from "./use-chat-store";

const STORE_KEY = "mirai-chat-store";
const lastUsedModel = { modelId: "vendor/model", modelName: "Model" };
const defaults = useChatStore.getState();

beforeEach(() => {
  useChatStore.setState(defaults, true);
  localStorage.clear();
});

it("saves the last used model and nothing else of the chat state", () => {
  useChatStore.getState().setChatModel("chat", lastUsedModel.modelId, lastUsedModel.modelName);

  expect(JSON.parse(localStorage.getItem(STORE_KEY)!).state).toEqual({ lastUsedModel });
});

it("restores the last used model after a restart", async () => {
  localStorage.setItem(STORE_KEY, JSON.stringify({ state: { lastUsedModel }, version: 0 }));

  await useChatStore.persist.rehydrate();

  expect(useChatStore.getState()).toMatchObject({ lastUsedModel, chatModels: {}, messages: [] });
});
