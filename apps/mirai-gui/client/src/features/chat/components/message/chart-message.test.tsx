import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAppStore } from "@/stores/use-app-store";
import { CHART_TYPES, type ChartSpec } from "@/types/chart";
import ChartMessage, { chartConfiguration } from "./chart-message";

const drawing = vi.hoisted(() => ({ create: vi.fn(), destroy: vi.fn() }));
vi.mock("chart.js/auto", () => ({
  default: class {
    constructor(canvas: unknown, config: unknown) {
      drawing.create(canvas, config);
    }
    destroy() {
      drawing.destroy();
    }
  },
}));

const chart: ChartSpec = {
  type: "bar",
  title: "<img src=x onerror=alert(1)>",
  labels: ["Rent", "Food"],
  datasets: [{ label: "Monthly spending", data: [900, 350] }],
};
beforeEach(() => {
  vi.clearAllMocks();
  useAppStore.setState({ isDarkMode: true });
});
afterEach(cleanup);

it("renders text safely and keeps completed charts stable through later streaming snapshots", () => {
  const view = render(<ChartMessage chart={chart} />);
  expect(screen.getByText(chart.title)).toBeTruthy();
  expect(view.container.querySelector("img")).toBeNull();
  expect(screen.getByRole("img").tagName).toBe("CANVAS");
  expect(drawing.create).toHaveBeenCalledTimes(1);
  view.rerender(<ChartMessage chart={structuredClone(chart)} />);
  expect(drawing.create).toHaveBeenCalledTimes(1);
  act(() => useAppStore.setState({ isDarkMode: false }));
  expect(drawing.create).toHaveBeenCalledTimes(2);
  expect(drawing.destroy).toHaveBeenCalledTimes(1);
  view.unmount();
  expect(drawing.destroy).toHaveBeenCalledTimes(2);
});

it("refuses arbitrary options even if a malformed chart reaches the renderer", () => {
  render(<ChartMessage chart={{ ...chart, options: { onClick: "alert(1)" } } as ChartSpec} />);
  expect(drawing.create).not.toHaveBeenCalled();
  expect(screen.getByText("This chart could not be displayed.")).toBeTruthy();
});

it("defaults charts to medium and applies each model-selected height", () => {
  const view = render(<ChartMessage chart={chart} />);
  expect(screen.getByRole("img").parentElement?.style.height).toBe("480px");
  for (const [height, pixels] of [
    ["small", 320],
    ["medium", 480],
    ["big", 640],
  ] as const) {
    view.rerender(<ChartMessage chart={{ ...chart, height }} />);
    expect(screen.getByRole("img").parentElement?.style.height).toBe(`${pixels}px`);
  }
  view.rerender(<ChartMessage chart={chart} />);
  expect(screen.getByRole("img").parentElement?.style.height).toBe("480px");
});

it.each(["pie", "doughnut", "polarArea"] as const)(
  "explains an all-zero %s while preserving its canvas and data",
  (type) => {
    const zeroChart: ChartSpec = {
      ...chart,
      type,
      datasets: [
        { label: "First", data: [0, 0] },
        { label: "Second", data: [0, 0] },
      ],
    };
    const view = render(<ChartMessage chart={zeroChart} />);
    const explanation = screen.getByText("All values are zero, so there are no segments to show.");
    expect(screen.getByRole("img").getAttribute("aria-describedby")).toBe(explanation.id);
    expect(screen.getAllByRole("cell").filter((cell) => cell.textContent === "0")).toHaveLength(4);
    expect(drawing.create.mock.calls[0]?.[1].data.datasets.map((dataset: { data: number[] }) => dataset.data)).toEqual([
      [0, 0],
      [0, 0],
    ]);

    view.rerender(
      <ChartMessage chart={{ ...zeroChart, datasets: [zeroChart.datasets[0]!, { label: "Second", data: [0, 1] }] }} />,
    );
    expect(screen.queryByText(explanation.textContent!)).toBeNull();
    expect(screen.getByRole("img").hasAttribute("aria-describedby")).toBe(false);
  },
);

it.each(["bar", "line", "radar"] as const)("does not treat valid zero values in a %s as missing segments", (type) => {
  render(<ChartMessage chart={{ ...chart, type, datasets: [{ label: "Data", data: [0, 0] }] }} />);
  expect(screen.queryByText("All values are zero, so there are no segments to show.")).toBeNull();
  expect(drawing.create).toHaveBeenCalledOnce();
});

it.each(CHART_TYPES)("builds %s without sharing mutable dataset objects with the saved transcript", (type) => {
  const xy = type === "scatter" || type === "bubble";
  const spec: ChartSpec = {
    ...chart,
    type,
    labels: xy ? undefined : chart.labels,
    datasets: [{ label: "Data", data: xy ? [{ x: 2, y: 4, ...(type === "bubble" ? { r: 8 } : {}) }] : [1, 2] }],
  };
  const config = chartConfiguration(spec, true);
  expect(config.type).toBe(type);
  expect(config.data.datasets[0]?.data).toEqual(spec.datasets[0]?.data);
  expect(config.data.datasets[0]?.data).not.toBe(spec.datasets[0]?.data);
  if (xy) expect(config.data.datasets[0]?.data[0]).not.toBe(spec.datasets[0]?.data[0]);
});
