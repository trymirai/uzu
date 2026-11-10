import { expect, it } from "vitest";
import { classifyReasoningSupport } from "./chat-support";

const capabilities = (reasoningEfforts: string[], supportsDisableReasoning = false, supportsReasoning = true) => ({
  supportsReasoning,
  supportsDisableReasoning,
  reasoningEfforts,
  supportsTools: false,
});

it("classifies reasoning support like the desktop catalog", () => {
  expect(classifyReasoningSupport(capabilities([], false, false))).toEqual({ kind: "unsupported" });
  expect(classifyReasoningSupport(capabilities(["Default"]))).toEqual({ kind: "alwaysOn" });
  expect(classifyReasoningSupport(capabilities(["Disabled", "Default"], true), "default")).toEqual({
    kind: "toggle",
    defaultEffort: "default",
  });
  // uzu lists efforts in template order; the client shows them from cheapest to strongest.
  expect(classifyReasoningSupport(capabilities(["High", "Low", "Disabled", "Medium"], true), "medium")).toEqual({
    kind: "levels",
    efforts: ["disabled", "low", "medium", "high"],
    defaultEffort: "medium",
  });
  expect(classifyReasoningSupport(capabilities(["XHigh", "Low"]), "default")).toEqual({
    kind: "levels",
    efforts: ["low", "xhigh"],
  });
});

it("does not offer an enabled mode that is not distinct from disabling reasoning", () => {
  expect(classifyReasoningSupport(capabilities(["Disabled"], true), "disabled")).toEqual({
    kind: "levels",
    efforts: ["disabled"],
    defaultEffort: "disabled",
  });
  expect(classifyReasoningSupport(capabilities(["Disabled", "Default"], true), undefined)).toEqual({
    kind: "levels",
    efforts: ["disabled"],
  });
});
