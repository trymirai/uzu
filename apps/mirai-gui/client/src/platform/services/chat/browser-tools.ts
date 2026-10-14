import { isChartSpec } from "@/types/chart";
import type { LlmRunParams } from "@/types/llm-stream";

export const TITLE_STYLE =
  "Write a natural, human-readable phrase with normal capitalization and spaces between words. Do not format it as a hashtag, an underscore-separated identifier, a hyphen-separated slug, or a CamelCase identifier. If the user explicitly supplies an exact name, preserve it.";
export const titlePrompt = (userText: string) =>
  `Give this chat a concise, descriptive title. ${TITLE_STYLE} Aim for about 25 characters so it fits in the sidebar. Reply with only the title, without quotes or explanation.\n\nMessage: ${JSON.stringify(userText)}`;
export function normalizeChatName(value: unknown): string {
  if (typeof value !== "string") throw new Error("Chat name must be a string");
  const title = value
    .trim()
    .split(/\s+/)
    .join(" ")
    .replace(/^["'`]+|["'`]+$/g, "")
    .trim();
  if (!title || Array.from(title).some((character) => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127))
    throw new Error("Chat names must not be empty or contain control characters");
  return title;
}

type ToolDefinition = { type: "function"; function: { name: string; description: string; parameters: object } };
export type ToolCall = { name: string; arguments: unknown };
const text = { type: "string", minLength: 1, maxLength: 200 };
const chartSchema = {
  type: "object",
  additionalProperties: false,
  required: ["type", "title", "datasets"],
  properties: {
    type: { type: "string", enum: ["bar", "line", "scatter", "bubble", "pie", "doughnut", "radar", "polarArea"] },
    title: text,
    height: { type: "string", enum: ["small", "medium", "big"] },
    xLabel: text,
    yLabel: text,
    labels: { type: "array", items: text, minItems: 1, maxItems: 500 },
    datasets: {
      type: "array",
      minItems: 1,
      maxItems: 8,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["label", "data"],
        properties: {
          label: text,
          data: {
            type: "array",
            minItems: 1,
            maxItems: 500,
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
                    r: { type: "number", minimum: 0, maximum: 100 },
                  },
                },
              ],
            },
          },
        },
      },
    },
  },
};
const definition = (
  name: string,
  description: string,
  properties: object = {},
  required: string[] = [],
): ToolDefinition => ({
  type: "function",
  function: { name, description, parameters: { type: "object", additionalProperties: false, properties, required } },
});
const parseObject = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
    throw new Error("Tool arguments must be an object");
  return parsed as Record<string, unknown>;
};
function chartArgument(value: unknown): unknown {
  const chart = parseObject(value);
  // Markup encodings pass JSON arguments as strings, including numeric leaves.
  const number = (value: unknown) =>
    typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value)) ? Number(value) : value;
  if (Array.isArray(chart.datasets))
    chart.datasets = chart.datasets.map((dataset) => {
      const row = parseObject(dataset);
      if (Array.isArray(row.data))
        row.data = row.data.map((value: unknown) => {
          if (value && typeof value === "object" && !Array.isArray(value))
            return Object.fromEntries(Object.entries(value).map(([key, n]) => [key, number(n)]));
          return number(value);
        });
      return row;
    });
  return chart;
}

export function browserTools(params: LlmRunParams, supported: boolean) {
  const tools: ToolDefinition[] = [];
  if (supported && params.dateTimeToolEnabled)
    tools.push(
      definition(
        "get_current_date_time",
        "Returns the current date and time in UTC and the device's local time, with explicit UTC offsets.",
      ),
    );
  if (supported && params.chartToolEnabled)
    tools.push(
      definition(
        "show_chart",
        "Display a chart inline in the conversation. Supply chart data only, never code, callbacks, plugins, URLs, or Chart.js options. Use numeric values and matching labels for categorical charts; use x/y points for scatter and x/y/r points for bubble. Optional height is small (320 px), medium (480 px, default), or big (640 px). The returned chart is displayed to the user.",
        { chart: chartSchema },
        ["chart"],
      ),
    );
  if (supported && params.modelChatNamingEnabled)
    tools.push(
      definition(
        "set_chat_name",
        "Set or update the name of the current chat.",
        {
          name: { type: "string", description: `A short, human-readable phrase. ${TITLE_STYLE}` },
          hidden: {
            type: "boolean",
            description:
              "True for autonomous housekeeping; false when the user explicitly requests naming or renaming.",
          },
        },
        ["name", "hidden"],
      ),
    );
  const names = new Set(tools.map((tool) => tool.function.name));
  const messages = params.messages.map((message) => ({ ...message }));
  if (names.has("set_chat_name")) {
    const instruction = `Use set_chat_name to give this chat a concise, descriptive name in your first turn. Update the name when the conversation's main topic changes or the user asks you to rename it. ${TITLE_STYLE} Aim for about 25 characters so the name fits in the sidebar. This is an approximate display budget, not a required length. If the tool warns that the name is too long, call it again with a shorter name before finishing this reply. Set hidden to true for autonomous naming housekeeping, and false when the user explicitly asks you to name or rename the chat. Answer the user's request normally; do not mention autonomous housekeeping or ask for permission to name the chat.`;
    const system = messages.find((message) => message.role === "system");
    if (system) system.content += `\n\n${instruction}`;
    else messages.unshift({ role: "system", content: instruction });
  }
  return {
    params: {
      ...params,
      messages,
      tools,
      dateTimeToolEnabled: supported && params.dateTimeToolEnabled,
      chartToolEnabled: supported && params.chartToolEnabled,
      modelChatNamingEnabled: supported && params.modelChatNamingEnabled,
    },
    execute(call: ToolCall) {
      try {
        if (!names.has(call.name)) throw new Error(`Unknown or disabled tool: ${call.name}`);
        const args = parseObject(call.arguments);
        switch (call.name) {
          case "get_current_date_time": {
            if (Object.keys(args).length) throw new Error("get_current_date_time takes no arguments");
            const now = new Date();
            const date = (timeZone?: string) =>
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
              }).format(now);
            return { value: `Current UTC date and time: ${date("UTC")}.\nDevice local date and time: ${date()}.` };
          }
          case "show_chart": {
            if (Object.keys(args).some((key) => key !== "chart")) throw new Error("show_chart accepts only chart");
            const chart = chartArgument(args.chart);
            if (!isChartSpec(chart))
              throw new Error(
                "Invalid chart: use the declared chart schema, finite numbers, matching labels, and at most 2,000 points",
              );
            return { value: chart, chart };
          }
          case "set_chat_name": {
            if (Object.keys(args).some((key) => key !== "name" && key !== "hidden"))
              throw new Error("set_chat_name accepts only name and hidden");
            if (
              typeof args.hidden !== "boolean" &&
              !(typeof args.hidden === "string" && /^(true|false)$/i.test(args.hidden.trim()))
            )
              throw new Error("hidden must be true or false");
            const name = normalizeChatName(args.name);
            const length = Array.from(name).length;
            return {
              chatName: name,
              value:
                length > 28
                  ? `Name saved, but may be clipped: it has ${length} characters, above the 28-character display budget. Aim for about 25 characters. Call set_chat_name again with a shorter phrase before finishing your reply; shorten the wording, keeping normal spaces.`
                  : "Chat name accepted.",
            };
          }
          default:
            throw new Error("Unknown tool");
        }
      } catch (error) {
        return { failed: true, value: { error: error instanceof Error ? error.message : String(error) } };
      }
    },
  };
}
