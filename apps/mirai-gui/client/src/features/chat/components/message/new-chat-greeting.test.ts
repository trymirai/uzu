import { afterEach, expect, it } from "vitest";
import { chooseGreeting, FIRST_GREETING, GREETINGS, rememberGreeting } from "./new-chat-greeting";

afterEach(() => localStorage.clear());

it("introduces itself the first time, and keeps doing so until a greeting has been shown", () => {
  expect(chooseGreeting().greeting).toEqual(FIRST_GREETING);
  expect(chooseGreeting().greeting).toEqual(FIRST_GREETING);
});

it("says something else once the introduction has been shown", () => {
  rememberGreeting(chooseGreeting());
  const next = chooseGreeting(() => 0);
  expect(next.index).toBe(0);
  expect(next.greeting).toEqual(GREETINGS[0]);
});

it("never repeats the line it said last", () => {
  rememberGreeting(chooseGreeting());
  rememberGreeting(chooseGreeting(() => 0));
  expect(chooseGreeting(() => 0).index).toBe(1);
});
