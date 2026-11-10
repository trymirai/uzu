import type { NativeTool } from "@uzu/wasm";
import { isChartSpec } from "@/types/chart";
import type { RunEvent } from "@/platform/services/chat/run-stream";
import type { UzuBindings } from "../uzu-runtime";

// Approximate prose budget for the sidebar's 186px label at 13px Inter.
// Glyph widths vary, so wider characters still rely on the UI's ellipsis.
export const TITLE_TARGET_LENGTH = 25;
const TITLE_SOFT_LIMIT = 28;
export const TITLE_STYLE_INSTRUCTION =
  "Write a natural, human-readable phrase with normal capitalization and spaces between words. " +
  "Do not format it as a hashtag, an underscore-separated identifier, a hyphen-separated slug, or a CamelCase identifier. " +
  "If the user explicitly supplies an exact name, preserve it.";

export const NAMING_INSTRUCTION =
  "Use set_chat_name to give this chat a concise, descriptive name in your first turn. " +
  "Update the name when the conversation's main topic changes or the user asks you to rename it. " +
  `${TITLE_STYLE_INSTRUCTION} ` +
  `Aim for about ${TITLE_TARGET_LENGTH} characters so the name fits in the sidebar. ` +
  "This is an approximate display budget, not a required length. " +
  "If the tool warns that the name is too long, call it again with a shorter name before finishing this reply. " +
  "Set hidden to true for autonomous naming housekeeping, and false when the user explicitly asks you to name or rename the chat. " +
  "Answer the user's request normally; do not mention autonomous housekeeping or ask for permission to name the chat.";

export const titlePrompt = (userText: string): string =>
  `Give this chat a concise, descriptive title. ${TITLE_STYLE_INSTRUCTION} Aim for about ${TITLE_TARGET_LENGTH} characters ` +
  `so it fits in the sidebar. Reply with only the title, without quotes or explanation.\n\nMessage: ${JSON.stringify(userText)}`;

// Rust's char::is_control: the C0 and C1 control ranges.
// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u001F\u007F-\u009F]/;

export function normalizeName(value: unknown): string {
  if (typeof value !== "string") throw new Error("Chat name must be a string");
  const name = value
    // Rust's split_whitespace also splits on U+0085 (NEL).
    .split(/[\s\u0085]+/)
    .filter(Boolean)
    .join(" ")
    .replace(/^["'`]+|["'`]+$/g, "")
    .trim();
  if (!name || CONTROL.test(name)) throw new Error("Chat names must not be empty or contain control characters");
  return name;
}

function titleLengthWarning(name: string): string | undefined {
  const length = Array.from(name).length;
  if (length <= TITLE_SOFT_LIMIT) return undefined;
  return (
    `Name saved, but may be clipped: it has ${length} characters, above the ${TITLE_SOFT_LIMIT}-character display budget. ` +
    `Aim for about ${TITLE_TARGET_LENGTH} characters. ` +
    "Call set_chat_name again with a shorter phrase before finishing your reply; shorten the wording, keeping normal spaces. " +
    "If you finish without replacing it, this name will be used and clipped in the interface as needed."
  );
}

/** The naming tool lives as long as its session; each run receives the names chosen during it. */
export class ChatNaming {
  private emit?: (event: RunEvent) => void;
  private name?: string;

  begin(emit: (event: RunEvent) => void): { name(): string | undefined; end(): void } {
    this.emit = emit;
    this.name = undefined;
    return {
      name: () => this.name,
      end: () => {
        if (this.emit === emit) this.emit = undefined;
      },
    };
  }

  setName(args: Record<string, unknown>): string {
    expectKeys(args, ["name", "hidden"]);
    // The transcript reads visibility from the recorded call.
    if (typeof args.hidden !== "boolean") throw new Error("hidden must be true or false");
    const name = normalizeName(args.name);
    if (!this.emit) throw new Error("No chat is currently running");
    this.emit({ type: "chatName", name });
    this.name = name;
    return titleLengthWarning(name) ?? "Chat name accepted.";
  }
}

const formatInstant = (now: Date, timeZone?: string): string => {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-GB", {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23",
      timeZone,
      timeZoneName: "longOffset",
    })
      .formatToParts(now)
      .map((part) => [part.type, part.value]),
  ) as Record<Intl.DateTimeFormatPartTypes, string>;
  // "GMT+04:00" (or "GMT" at zero offset) becomes "UTC+04:00", as on desktop.
  const offset = parts.timeZoneName.replace(/^GMT/, "") || "+00:00";
  return `${parts.weekday}, ${parts.day} ${parts.month} ${parts.year} at ${parts.hour}:${parts.minute}:${parts.second} (UTC${offset})`;
};

export const currentDateTime = (now = new Date()): string =>
  `Current UTC date and time: ${formatInstant(now, "UTC")}.\nDevice local date and time: ${formatInstant(now)}.`;

function expectKeys(args: Record<string, unknown>, allowed: string[]) {
  const unknown = Object.keys(args).find((key) => !allowed.includes(key));
  if (unknown !== undefined) throw new Error(`Unknown argument: ${unknown}`);
}

const parseObject = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
    throw new Error("Tool arguments must be an object");
  return parsed as Record<string, unknown>;
};

export function showChart(args: Record<string, unknown>): unknown {
  expectKeys(args, ["chart"]);
  const chart = parseObject(args.chart);
  if (!isChartSpec(chart))
    throw new Error(
      "Invalid chart: use the declared chart schema, finite numbers, matching labels, and at most 2,000 points",
    );
  return chart;
}

const text = { type: "string", minLength: 1, maxLength: 200 };
// Keep aligned with ChartSpec in desktop/src/chat/chart.rs and isChartSpec.
const CHART_SCHEMA = {
  type: "object",
  description: "Data for an inline chart. Only the fields in this schema are accepted.",
  additionalProperties: false,
  required: ["type", "title", "datasets"],
  properties: {
    type: { type: "string", enum: ["bar", "line", "scatter", "bubble", "pie", "doughnut", "radar", "polarArea"] },
    title: { ...text, description: "A concise chart title, at most 200 characters." },
    height: {
      type: "string",
      enum: ["small", "medium", "big"],
      description: "Chart height: small (320 px), medium (480 px, the default), or big (640 px).",
    },
    labels: {
      type: "array",
      items: { type: "string" },
      minItems: 1,
      maxItems: 500,
      description: "Category names, required for every chart except scatter and bubble.",
    },
    datasets: {
      type: "array",
      minItems: 1,
      maxItems: 8,
      description: "One to eight series, with at most 2,000 data points in the whole chart.",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["label", "data"],
        properties: {
          label: { ...text, description: "A concise series name, at most 200 characters." },
          data: {
            type: "array",
            minItems: 1,
            maxItems: 500,
            description: "Numbers matching labels for categorical charts, or x/y points for scatter and bubble.",
            items: {
              anyOf: [
                { type: "number" },
                {
                  type: "object",
                  additionalProperties: false,
                  required: ["x", "y"],
                  properties: {
                    x: { type: "number" },
                    y: { type: "number" },
                    r: {
                      type: "number",
                      minimum: 0,
                      maximum: 100,
                      description: "Bubble radius in pixels, from 0 to 100. Required for bubble, omitted for scatter.",
                    },
                  },
                },
              ],
            },
          },
        },
      },
    },
    xLabel: { ...text, description: "Optional horizontal axis label, at most 200 characters." },
    yLabel: { ...text, description: "Optional vertical axis label, at most 200 characters." },
  },
};

const objectSchema = (properties: object = {}, required: string[] = []) => ({
  type: "object",
  additionalProperties: false,
  properties,
  required,
});

type ToolSpec = {
  name: string;
  description: string;
  parameters: object;
  run(args: Record<string, unknown>): unknown;
};

export type ToolFlags = { dateTime: boolean; chart: boolean; naming: ChatNaming | null };

export function toolSpecs({ dateTime, chart, naming }: ToolFlags): ToolSpec[] {
  const specs: ToolSpec[] = [];
  if (dateTime)
    specs.push({
      name: "get_current_date_time",
      description: "Returns the current date and time in UTC and the device's local time, with explicit UTC offsets.",
      parameters: objectSchema(),
      run: (args) => {
        expectKeys(args, []);
        return currentDateTime();
      },
    });
  if (chart)
    specs.push({
      name: "show_chart",
      description:
        "Display a chart inline in the conversation. Supply chart data only, never code, callbacks, plugins, URLs, or Chart.js options. " +
        "Use numeric values and matching labels for categorical charts; use x/y points for scatter and x/y/r points for bubble. " +
        "Optional height is small (320 px), medium (480 px, default), or big (640 px). The returned chart is displayed to the user.",
      parameters: objectSchema({ chart: CHART_SCHEMA }, ["chart"]),
      run: showChart,
    });
  if (naming)
    specs.push({
      name: "set_chat_name",
      description: "Set or update the name of the current chat.",
      parameters: objectSchema(
        {
          name: {
            type: "string",
            description:
              "A short, human-readable phrase with normal capitalization and spaces between words. " +
              "Avoid hashtags, underscore-separated identifiers, hyphen-separated slugs, and CamelCase identifiers. " +
              "Preserve an explicitly requested exact name; otherwise aim to fit the provided character budget.",
          },
          hidden: {
            type: "boolean",
            description:
              "True for autonomous housekeeping; false when the user explicitly requests naming or renaming.",
          },
        },
        ["name", "hidden"],
      ),
      run: (args) => naming.setName(args),
    });
  return specs;
}

/** Wraps the specs as uzu tools. A thrown error becomes the call's `{"error": …}` result. */
export const nativeTools = (api: UzuBindings, specs: ToolSpec[]): NativeTool[] =>
  specs.map(
    (spec) =>
      new api.NativeTool(
        new api.ToolFunction(spec.name, spec.description, new api.Value(JSON.stringify(spec.parameters)), null),
        async (argumentsJson: string) => JSON.stringify(spec.run(parseObject(argumentsJson)) ?? null),
        () => {},
      ),
  );
