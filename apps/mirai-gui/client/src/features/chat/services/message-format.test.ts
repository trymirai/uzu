import { expect, it } from "vitest";
import { Roles } from "@/types/chat";
import { normalizeMessagesForRun } from "./message-format";

it("keeps both texts when two user turns follow each other", () => {
  const out = normalizeMessagesForRun([
    { role: Roles.User, content: "first question" },
    { role: Roles.User, content: "a follow-up" },
  ]);

  expect(out).toEqual([{ role: Roles.User, content: "first question\n\na follow-up" }]);
});

it("drops empty assistant turns and keeps alternation", () => {
  const out = normalizeMessagesForRun([
    { role: Roles.User, content: "q" },
    { role: Roles.Assistant, content: "" },
    { role: Roles.Assistant, content: "a" },
    { role: Roles.User, content: "q2" },
  ]);

  expect(out.map((m) => m.role)).toEqual([Roles.User, Roles.Assistant, Roles.User]);
});
