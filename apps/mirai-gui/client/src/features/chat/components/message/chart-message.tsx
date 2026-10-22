import Chart from "chart.js/auto";
import type { ChartConfiguration } from "chart.js";
import { memo, useEffect, useId, useRef } from "react";
import { useAppStore } from "@/stores/use-app-store";
import { isChartSpec, type ChartSpec } from "@/types/chart";
import { chartTable } from "@/utils/chart-data";

const CHART_HEIGHT_PX = { small: 320, medium: 480, big: 640 };

export function chartConfiguration(spec: ChartSpec, dark: boolean): ChartConfiguration {
  const colors = dark
    ? ["#74a9ff", "#f4ab72", "#80c99a", "#b59aef", "#ed94b9", "#69cbd2", "#d6ca73", "#abb8d1"]
    : ["#3676cf", "#bd6828", "#32824e", "#7950b6", "#b84477", "#25828a", "#8f7c1d", "#60718e"];
  const foreground = dark ? "#c5c7cb" : "#43464b";
  const grid = dark ? "#ffffff18" : "#00000018";
  const font = { family: "InterVariable, Inter, system-ui, sans-serif", size: 12 };
  const tooltipBackground = dark ? "#191919" : "#f0f0f0";
  const tooltipForeground = dark ? "#eeeeee" : "#202020";
  const segments = spec.type === "pie" || spec.type === "doughnut" || spec.type === "polarArea";
  const radial = spec.type === "radar" || spec.type === "polarArea";
  const cartesian = !segments && !radial;
  return {
    type: spec.type,
    data: {
      labels: spec.labels ? [...spec.labels] : undefined,
      datasets: spec.datasets.map((dataset, index) => {
        const color = colors[index % colors.length]!;
        return {
          label: dataset.label,
          // Copy only known data fields. No model-supplied options or callbacks
          // are passed into Chart.js, and Chart.js cannot mutate saved data.
          data: dataset.data.map((point) =>
            typeof point === "number"
              ? point
              : { x: point.x, y: point.y, ...(point.r !== undefined ? { r: point.r } : {}) },
          ),
          borderColor: segments ? colors : color,
          backgroundColor: segments ? colors.map((entry) => `${entry}b3`) : `${color}66`,
          borderWidth: 2,
          pointRadius: spec.type === "line" ? 2 : undefined,
          fill: spec.type === "radar",
        };
      }),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      color: foreground,
      font,
      plugins: {
        colors: { enabled: false },
        legend: {
          display: segments || spec.datasets.length > 1,
          position: "bottom",
          labels: { color: foreground, font, boxWidth: 12, boxHeight: 12, padding: 16 },
        },
        tooltip: {
          backgroundColor: tooltipBackground,
          titleColor: tooltipForeground,
          bodyColor: tooltipForeground,
          footerColor: tooltipForeground,
          titleFont: { ...font, weight: "bold" },
          bodyFont: font,
          footerFont: { ...font, weight: "bold" },
          borderColor: dark ? "#ffffff1a" : "#0000000f",
          borderWidth: 1,
          multiKeyBackground: tooltipBackground,
        },
      },
      ...(cartesian
        ? {
            scales: {
              x: {
                grid: { color: grid },
                border: { color: grid },
                ticks: { color: foreground, font },
                title: { display: !!spec.xLabel, text: spec.xLabel, color: foreground, font },
              },
              y: {
                beginAtZero: spec.type === "bar",
                grid: { color: grid },
                border: { color: grid },
                ticks: { color: foreground, font },
                title: { display: !!spec.yLabel, text: spec.yLabel, color: foreground, font },
              },
            },
          }
        : radial
          ? {
              scales: {
                r: {
                  grid: { color: grid },
                  angleLines: { color: grid },
                  pointLabels: { color: foreground, font: { ...font, size: 10 } },
                  ticks: { color: foreground, font, backdropColor: "transparent" },
                },
              },
            }
          : {}),
    },
  };
}

function ChartMessage({ chart }: { chart: ChartSpec }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const emptyDescriptionId = useId();
  const dark = useAppStore((state) => state.isDarkMode);
  const valid = isChartSpec(chart);
  useEffect(() => {
    if (!valid || !canvasRef.current) return;
    const instance = new Chart(canvasRef.current, chartConfiguration(chart, dark));
    return () => instance.destroy();
  }, [chart, dark, valid]);

  if (!valid) return <p className="my-3 text-[13px] text-label-muted">This chart could not be displayed.</p>;
  const table = chartTable(chart);
  const allSegmentsZero =
    (chart.type === "pie" || chart.type === "doughnut" || chart.type === "polarArea") &&
    chart.datasets.every((dataset) => dataset.data.every((value) => value === 0));
  return (
    <figure className="my-3 min-w-0 rounded-lg border border-border-default bg-surface-elevated p-4">
      <figcaption className="mb-3 text-[14px] font-medium text-text-primary [overflow-wrap:anywhere]">
        {chart.title}
      </figcaption>
      <div className="relative min-w-0" style={{ height: CHART_HEIGHT_PX[chart.height ?? "medium"] }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`${chart.title} (${chart.type} chart)`}
          aria-describedby={allSegmentsZero ? emptyDescriptionId : undefined}
        >
          <table>
            <thead>
              <tr>
                {table.headers.map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, index) => (
                <tr key={index}>
                  {row.map((cell, column) => (
                    <td key={column}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </canvas>
        {allSegmentsZero && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
            <p
              id={emptyDescriptionId}
              className="rounded-md bg-surface-elevated px-3 py-2 text-center text-[13px] text-text-secondary"
            >
              All values are zero, so there are no segments to show.
            </p>
          </div>
        )}
      </div>
    </figure>
  );
}

// Streaming snapshots repeat completed charts. Do not redraw them for every
// following token; theme changes still update through the store subscription.
export default memo(
  ChartMessage,
  (previous, next) => previous.chart === next.chart || JSON.stringify(previous.chart) === JSON.stringify(next.chart),
);
