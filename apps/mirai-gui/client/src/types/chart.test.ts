import { expect, it } from "vitest";
import { CHART_HEIGHTS, CHART_TYPES, isChartSpec, type ChartSpec } from "./chart";

const chart: ChartSpec = {
  type: "bar",
  title: "Budget",
  labels: ["Rent", "Food"],
  datasets: [{ label: "Monthly spending", data: [900, 350] }],
};

it.each(CHART_TYPES)("accepts the data shape for %s charts", (type) => {
  const xy = type === "scatter" || type === "bubble";
  expect(
    isChartSpec({
      ...chart,
      type,
      labels: xy ? undefined : chart.labels,
      datasets: [{ label: "Example", data: xy ? [{ x: 2, y: 4, ...(type === "bubble" ? { r: 6 } : {}) }] : [1, 2] }],
    }),
  ).toBe(true);
});

it.each(CHART_HEIGHTS)("accepts the optional %s chart height", (height) => {
  expect(isChartSpec({ ...chart, height })).toBe(true);
});

it.each([
  { type: "custom" },
  { height: "large" },
  { height: "320px" },
  { height: 320 },
  { height: null },
  { options: { onClick: "alert(1)" } },
  { plugins: [{ id: "code", beforeDraw: "alert(1)" }] },
  { title: "   " },
  { title: "x".repeat(201) },
  { labels: [] },
  { labels: ["Rent"] },
  { datasets: [{ label: "Bad", data: [NaN, 1] }] },
  { datasets: [{ label: "Bad", data: [Infinity, 1] }] },
  { datasets: [{ label: "Bad", data: ["1", 2] }] },
  { datasets: [{ label: "Bad", data: [1, 2], parsing: "anything" }] },
  {
    datasets: [
      {
        label: "Bad",
        data: [
          { x: 1, y: 2 },
          { x: 2, y: 3 },
        ],
      },
    ],
  },
  { type: "pie", datasets: [{ label: "Bad", data: [-1, 2] }] },
  { labels: Array(501).fill("x"), datasets: [{ label: "Too long", data: Array(501).fill(1) }] },
  { datasets: Array(9).fill(chart.datasets[0]) },
  { labels: Array(500).fill("x"), datasets: Array(5).fill({ label: "Too many", data: Array(500).fill(1) }) },
])("rejects unsupported configuration or invalid data: %j", (override) => {
  expect(isChartSpec({ ...chart, ...override })).toBe(false);
});

it("validates XY points and bubble radii without accepting point options", () => {
  const pointChart = (type: "bubble" | "scatter", point: unknown) => ({
    type,
    title: "Points",
    datasets: [{ label: "Series", data: [point] }],
  });
  expect(isChartSpec(pointChart("bubble", { x: 1, y: 2 }))).toBe(false);
  expect(isChartSpec(pointChart("bubble", { x: 1, y: 2, r: 101 }))).toBe(false);
  expect(isChartSpec(pointChart("bubble", { x: 1, y: 2, r: -1 }))).toBe(false);
  expect(isChartSpec(pointChart("scatter", { x: 1, y: 2, r: 1 }))).toBe(false);
  expect(isChartSpec(pointChart("scatter", { x: 1, y: 2, onClick: "code" }))).toBe(false);
  expect(isChartSpec({ ...pointChart("scatter", { x: 1, y: 2 }), labels: ["Unexpected"] })).toBe(false);
});

it("treats markup and code-looking labels as plain text, and counts Unicode characters consistently", () => {
  expect(isChartSpec({ ...chart, title: "<img src=x onerror=alert(1)>" })).toBe(true);
  expect(isChartSpec({ ...chart, title: "😀".repeat(200) })).toBe(true);
  expect(isChartSpec({ ...chart, title: "😀".repeat(201) })).toBe(false);
});
