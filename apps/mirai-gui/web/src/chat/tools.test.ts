import { expect, it } from "vitest";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import {
  ChatNaming,
  currentDateTime,
  NAMING_INSTRUCTION,
  normalizeName,
  TITLE_TARGET_LENGTH,
  toolSpecs,
  type ToolFlags,
} from "./tools";

// Ported from the tests in desktop/src/chat/naming.rs and desktop/src/chat/session.rs.

// TITLE_SOFT_LIMIT is private; the Rust tests use it as 28.
const TITLE_SOFT_LIMIT = 28;
const ACCEPTED = "Chat name accepted.";

const findTool = (flags: Partial<ToolFlags>, name: string) => {
  const spec = toolSpecs({ dateTime: false, chart: false, naming: null, ...flags }).find((spec) => spec.name === name);
  if (!spec) throw new Error(`missing tool ${name}`);
  return spec;
};
// ChatNaming::tool(): the set_chat_name spec that the session registers.
const namingTool = (naming: ChatNaming) => findTool({ naming }, "set_chat_name");
const args = (name: string) => ({ name, hidden: true });
const capture = (naming: ChatNaming) => {
  const received: RunEvent[] = [];
  return { run: naming.begin((event) => received.push(event)), received };
};
// title_length_warning is private: the tool returns it in place of the acceptance message.
const warning = (name: string): string | undefined => {
  const naming = new ChatNaming();
  const { run } = capture(naming);
  const result = naming.setName(args(name));
  run.end();
  return result === ACCEPTED ? undefined : result;
};

it("does not treat the display budget as a hard limit when normalizing names", () => {
  expect(normalizeName("  `Exploring\n  London` ")).toBe("Exploring London");
  expect(() => normalizeName('" "')).toThrow();
  expect(normalizeName("X")).toBe("X");
  expect(() => normalizeName("x".repeat(TITLE_SOFT_LIMIT + 1))).not.toThrow();
  expect(() => normalizeName("x".repeat(51))).not.toThrow();
  expect(() => normalizeName("bad\0title")).toThrow();
  expect(warning("😀".repeat(TITLE_SOFT_LIMIT))).toBeUndefined();
  const text = warning("x".repeat(TITLE_SOFT_LIMIT + 3));
  expect(text).toContain(`has ${TITLE_SOFT_LIMIT + 3} characters`);
  expect(text).toContain("before finishing your reply");
});

it("allows some room above the title target", () => {
  expect(TITLE_TARGET_LENGTH).toBe(25);
  expect(NAMING_INSTRUCTION).toContain("Aim for about 25 characters");
  expect(warning("x".repeat(25))).toBeUndefined();
  expect(warning("😀".repeat(28))).toBeUndefined();
  const text = warning("😀".repeat(29));
  expect(text).toContain("has 29 characters");
  expect(text).toContain("above the 28-character display budget");
  expect(text).toContain("Aim for about 25 characters");
});

it("names only the active run and requires a name and visibility", () => {
  const naming = new ChatNaming();
  const tool = namingTool(naming);
  expect(() => tool.run(args("Before run"))).toThrow();
  const { run, received } = capture(naming);
  expect(() => tool.run(args(""))).toThrow();
  expect(run.name()).toBeUndefined();
  for (const invalid of [{ hidden: true }, { name: "Name" }, { name: "Name", hidden: "maybe" }]) {
    expect(() => tool.run(invalid)).toThrow();
  }
  tool.run(args("  First\nname "));
  tool.run({ name: "Changed name", hidden: false });
  expect(run.name()).toBe("Changed name");
  expect(received).toHaveLength(2);
  expect(received[0]).toEqual({ type: "chatName", name: "First name" });
  run.end();
  expect(() => tool.run(args("After run"))).toThrow();
  const next = naming.begin(() => {});
  expect(next.name()).toBeUndefined();
});

it("keeps over-budget names replaceable across tool continuations", () => {
  const naming = new ChatNaming();
  const tool = namingTool(naming);
  const { run, received } = capture(naming);
  const tooLong = "x".repeat(TITLE_SOFT_LIMIT + 1);
  const feedback = tool.run(args(tooLong));
  expect(typeof feedback).toBe("string");
  expect(feedback).toContain("Call set_chat_name again");
  expect(received).toHaveLength(1);
  expect(run.name()).toBe(tooLong);
  // A second automatic tool continuation is still inside the same run.
  tool.run(args("Shorter name"));
  expect(run.name()).toBe("Shorter name");
  run.end();
  expect(received).toHaveLength(2);
});

it("keeps the latest over-budget name without truncating it", () => {
  const naming = new ChatNaming();
  const tool = namingTool(naming);
  const { run, received } = capture(naming);
  tool.run(args("Initial name"));
  tool.run(args("x".repeat(TITLE_SOFT_LIMIT + 2)));
  const latest = "x".repeat(TITLE_SOFT_LIMIT + 1);
  tool.run(args(latest));
  expect(run.name()).toBe(latest);
  run.end();
  expect(received).toHaveLength(3);
});

it("publishes accepted names before a run is interrupted", () => {
  const naming = new ChatNaming();
  const tool = namingTool(naming);
  const { run, received } = capture(naming);
  const tooLong = "x".repeat(TITLE_SOFT_LIMIT + 1);
  tool.run(args(tooLong));
  // A client may stop accepting events immediately after cancellation.
  expect(received).toHaveLength(1);
  run.end();
  expect(received).toHaveLength(1);
  expect(received[0]).toEqual({ type: "chatName", name: tooLong });
  expect(() => tool.run(args("After run"))).toThrow();
});

it("requires hidden in the naming tool schema and accepts the coerced boolean", () => {
  const naming = new ChatNaming();
  const tool = namingTool(naming);
  const schema = tool.parameters as { required: string[]; properties: Record<string, { type: string }> };
  for (const field of ["name", "hidden"]) expect(schema.required).toContain(field);
  // The Rust test sends hidden: " TRUE ". uzu's ToolDescriptor (which wraps the
  // wasm NativeTool too) coerces string booleans by this schema type before the
  // callback runs, so the TS tool receives the coerced value.
  expect(schema.properties.hidden?.type).toBe("boolean");
  const run = naming.begin(() => {});
  tool.run({ name: "Markup name", hidden: true });
  expect(run.name()).toBe("Markup name");
});

const DAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday";
const MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December";
// "%A, %-d %B %Y at %H:%M:%S (UTC%:z)" from desktop/src/chat/session.rs.
const INSTANT = new RegExp(
  `^(${DAYS}), ([1-9]\\d?) (${MONTHS}) (\\d{4}) at (\\d{2}):(\\d{2}):(\\d{2}) \\(UTC([+-])(\\d{2}):(\\d{2})\\)\\.$`,
);

/** Parses one line of the tool's output into epoch seconds and its UTC offset in minutes. */
function parseInstant(line: string, prefix: string): { seconds: number; offset: number } {
  expect(line.startsWith(prefix)).toBe(true);
  const match = INSTANT.exec(line.slice(prefix.length));
  if (!match) throw new Error(`unexpected date-time format: ${line}`);
  const [, weekday, day, month, year, hour, minute, second, sign, offsetHours, offsetMinutes] = match;
  const offset = (sign === "-" ? -1 : 1) * (Number(offsetHours) * 60 + Number(offsetMinutes));
  const wall = Date.UTC(
    Number(year),
    MONTHS.split("|").indexOf(month!),
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  );
  // chrono's parse_from_str rejects a weekday that does not match the date.
  expect(DAYS.split("|")[(new Date(wall).getUTCDay() + 6) % 7]).toBe(weekday);
  return { seconds: wall / 1000 - offset * 60, offset };
}

function parseDateTime(value: string) {
  const lines = value.split("\n");
  expect(lines).toHaveLength(2);
  return {
    utc: parseInstant(lines[0]!, "Current UTC date and time: "),
    local: parseInstant(lines[1]!, "Device local date and time: "),
  };
}

it("returns the same current instant in UTC and local time", () => {
  const tool = findTool({ dateTime: true }, "get_current_date_time");
  expect(tool.name).toBe("get_current_date_time");
  const before = Math.floor(Date.now() / 1000);
  const value = tool.run({});
  const after = Math.floor(Date.now() / 1000);
  expect(typeof value).toBe("string");
  const { utc, local } = parseDateTime(value as string);
  expect(utc.seconds).toBe(local.seconds);
  expect(utc.seconds).toBeGreaterThanOrEqual(before);
  expect(utc.seconds).toBeLessThanOrEqual(after);
  expect(utc.offset).toBe(0);

  // A fixed instant checks the exact format and the device's offset at that instant.
  const now = new Date(Date.UTC(2026, 0, 5, 3, 4, 5, 678));
  const fixed = currentDateTime(now);
  expect(fixed.split("\n")[0]).toBe("Current UTC date and time: Monday, 5 January 2026 at 03:04:05 (UTC+00:00).");
  const parsed = parseDateTime(fixed);
  expect(parsed.utc.seconds).toBe(Math.floor(now.getTime() / 1000));
  expect(parsed.local.seconds).toBe(parsed.utc.seconds);
  // 0 - x avoids -0 at a zero offset.
  expect(parsed.local.offset).toBe(0 - now.getTimezoneOffset());
});
