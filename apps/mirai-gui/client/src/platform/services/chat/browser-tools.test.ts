import { expect, it } from "vitest";
import { browserTools, normalizeChatName, titlePrompt } from "./browser-tools";
const params = {
  repoId: "model",
  messages: [{ role: "user" as const, content: "Hello" }],
  dateTimeToolEnabled: true,
  chartToolEnabled: true,
  modelChatNamingEnabled: true,
};

it("advertises only enabled tools and executes device-local date/time", () => {
  const tools = browserTools(params, true);
  expect(tools.params.tools.map((tool) => tool.function.name)).toEqual([
    "get_current_date_time",
    "show_chart",
    "set_chat_name",
  ]);
  expect(tools.params.messages[0]?.role).toBe("system");
  expect(tools.execute({ name: "get_current_date_time", arguments: {} }).value).toContain("Current UTC date and time:");
  expect(tools.execute({ name: "get_current_date_time", arguments: {} }).value).toContain(
    "Device local date and time:",
  );
  expect(browserTools(params, false).params.tools).toEqual([]);
  expect(browserTools(params, false).params.dateTimeToolEnabled).toBe(false);
  expect(
    browserTools({ ...params, chartToolEnabled: false }, true).execute({ name: "show_chart", arguments: {} }),
  ).toMatchObject({ failed: true });
});
it("validates chart arguments including markup-encoded JSON and numbers", () => {
  const tools = browserTools(params, true);
  const chart = {
    type: "bar",
    title: "Sales",
    labels: ["May", "June"],
    datasets: [{ label: "Sales", data: ["2", "4"] }],
  };
  expect(tools.execute({ name: "show_chart", arguments: { chart: JSON.stringify(chart) } })).toMatchObject({
    chart: { datasets: [{ data: [2, 4] }] },
  });
  expect(
    tools.execute({ name: "show_chart", arguments: { chart: { ...chart, options: { onClick: "evil()" } } } }),
  ).toMatchObject({ failed: true });
  expect(tools.execute({ name: "show_chart", arguments: { chart: { ...chart, labels: ["May"] } } })).toMatchObject({
    failed: true,
  });
});
it("uses the desktop title instructions and validates naming tool results", () => {
  expect(titlePrompt('One\n"two"')).toContain('Message: "One\\n\\"two\\""');
  expect(titlePrompt("hello")).toContain("Aim for about 25 characters");
  expect(normalizeChatName(' "Good   title" ')).toBe("Good title");
  expect(() => normalizeChatName("bad\0title")).toThrow();
  expect(
    browserTools(params, true).execute({ name: "set_chat_name", arguments: { name: "Good title", hidden: "True" } }),
  ).toMatchObject({ chatName: "Good title", value: "Chat name accepted." });
  expect(
    browserTools(params, true).execute({ name: "set_chat_name", arguments: { name: "Good title", hidden: "maybe" } }),
  ).toMatchObject({ failed: true });
});
it("does not execute arbitrary model-supplied tool names", () => {
  expect(
    browserTools(params, true).execute({ name: "fetch", arguments: { url: "https://example.com" } }),
  ).toMatchObject({ failed: true });
});
