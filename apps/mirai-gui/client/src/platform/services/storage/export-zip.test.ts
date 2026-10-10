import JSZip from "jszip";
import { beforeEach, expect, it, vi } from "vitest";
import type { ChatData } from ".";
import type { ChartSpec } from "@/types/chart";
import { serializeToMarkdown } from "./markdown/serialize";
import { buildChatsZip } from "./export-zip";

const files = vi.hoisted(() => ({ list: vi.fn(), load: vi.fn() }));
vi.mock("./chat-files", () => ({ listChatFiles: files.list, loadChatFile: files.load }));

const budget: ChartSpec = {
  type: "bar",
  title: "Budget <!-- END_CONTENT -->",
  labels: ["Food | drink", "Transport\ntravel"],
  datasets: [
    { label: "Planned", data: [300, 100] },
    { label: "Actual", data: [280, 120] },
  ],
};
const points: ChartSpec = {
  type: "bubble",
  title: "Comparison",
  xLabel: "Cost",
  yLabel: "Value",
  datasets: [
    { label: "First", data: [{ x: 1.5, y: -2, r: 8 }] },
    { label: "Second", data: [{ x: 3, y: 4, r: 10 }] },
  ],
};
const timestamp = 1_700_000_000_000;
const chat: ChatData = {
  metadata: { id: "chat", title: "Charts", messageCount: 2, createdAt: timestamp, updatedAt: timestamp },
  messages: [
    { id: "u", sender: "user", text: "Plot these values", timestamp },
    {
      id: "a",
      sender: "assistant",
      text: "Summary of the data.",
      timestamp,
      output: {
        text: { parsed: { chainOfThought: "Compare the values", response: "Summary of the data." } },
        transcript: [
          { type: "chart", chart: budget },
          { type: "text", text: "Summary of the data." },
          { type: "chart", chart: points },
        ],
      },
    },
  ],
};

beforeEach(() => vi.resetAllMocks());

const exported = async (markdown: string): Promise<string> => {
  files.list.mockResolvedValue(["chat.md"]);
  files.load.mockResolvedValue(markdown);
  const zip = await JSZip.loadAsync((await buildChatsZip())!);
  return zip.file("chat.md")!.async("string");
};

it("exports serialized chart data as readable tables without losing text or interpreting label markup", async () => {
  const saved = serializeToMarkdown(chat);
  const markdown = await exported(saved);
  expect(markdown).toContain("**Budget &lt;!-- END\\_CONTENT --&gt;**");
  expect(markdown).toContain("| Category | Planned | Actual |\n| --- | --- | --- |");
  expect(markdown).toContain("| Food \\| drink | 300 | 280 |");
  expect(markdown).toContain("| Transport<br>travel | 100 | 120 |");
  expect(markdown).toContain("| Series | Cost | Value | Radius (px) |");
  expect(markdown).toContain("| First | 1.5 | -2 | 8 |");
  expect(markdown).toContain("| Second | 3 | 4 | 10 |");
  expect(markdown.indexOf("**Budget")).toBeLessThan(markdown.indexOf("**Comparison"));
  expect(markdown).toContain("Plot these values");
  expect(markdown).toContain("<summary>Thinking</summary>");
  expect(markdown).toContain("Compare the values");
  expect(markdown.match(/Summary of the data\./g)).toHaveLength(1);
  expect(markdown).not.toContain("<!--");
  expect(saved).toBe(serializeToMarkdown(chat));
});

it("exports charts in saved response versions and marks invalid chart data visibly", async () => {
  const assistant = chat.messages[1]!;
  const versioned: ChatData = {
    ...chat,
    messages: [
      {
        ...assistant,
        currentVersionIndex: 1,
        versions: [
          {
            id: "v1",
            modelId: "model",
            modelName: "Model",
            timestamp,
            text: "First version",
            output: { transcript: [{ type: "chart", chart: budget }] },
          },
          {
            id: "v2",
            modelId: "model",
            modelName: "Model",
            timestamp,
            text: "Second version",
            output: { transcript: [{ type: "chart", chart: points }] },
          },
        ],
      },
    ],
  };
  const markdown = await exported(serializeToMarkdown(versioned));
  expect(markdown.indexOf("**Budget")).toBeLessThan(markdown.indexOf("#### Version 2"));
  expect(markdown.indexOf("**Comparison")).toBeGreaterThan(markdown.indexOf("#### Version 2"));
  const corrupted = serializeToMarkdown(chat).replace('"type":"bar"', '"type":"unsupported"');
  expect(await exported(corrupted)).toContain("This chart could not be exported: invalid data.");
});

it.each(["response", "reasoning", "error"] as const)(
  "does not export chart metadata examples in %s or version content as actual charts",
  async (location) => {
    const example = `Example metadata:\n<!-- TRANSCRIPT: ${JSON.stringify([{ type: "chart", chart: points }])} -->`;
    const text = location === "response" ? example : "Original response";
    const output =
      location === "reasoning" ? { text: { parsed: { response: text, chainOfThought: example } } } : undefined;
    const exampleMessage = {
      id: "example",
      sender: "assistant" as const,
      timestamp,
      text,
      output,
      ...(location === "error" ? { error: example } : {}),
    };
    const saved = serializeToMarkdown({
      ...chat,
      messages: [
        exampleMessage,
        {
          ...exampleMessage,
          id: "versioned",
          currentVersionIndex: 0,
          versions: [{ ...exampleMessage, modelId: "model", modelName: "Model" }],
        },
        { ...exampleMessage, id: "actual-chart", output: { transcript: [{ type: "chart", chart: budget }] } },
      ],
    });
    const markdown = await exported(saved);
    expect(markdown).not.toContain("**Comparison**");
    expect(markdown).not.toContain("| First | 1.5 | -2 | 8 |");
    expect(markdown.match(/\*\*Budget/g)).toHaveLength(1);
  },
);
