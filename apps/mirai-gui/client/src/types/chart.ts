export const CHART_TYPES = ["bar", "line", "scatter", "bubble", "pie", "doughnut", "radar", "polarArea"] as const;
export type ChartKind = (typeof CHART_TYPES)[number];
export const CHART_HEIGHTS = ["small", "medium", "big"] as const;
export type ChartHeight = (typeof CHART_HEIGHTS)[number];
export type ChartPoint = { x: number; y: number; r?: number };
export type ChartSpec = {
  type: ChartKind;
  title: string;
  height?: ChartHeight;
  labels?: string[];
  datasets: Array<{ label: string; data: Array<number | ChartPoint> }>;
  xLabel?: string;
  yLabel?: string;
};

const objectWithKeys = (value: unknown, keys: string[]): value is Record<string, unknown> =>
  value !== null &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  Object.keys(value).every((key) => keys.includes(key));

const text = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0 && Array.from(value).length <= 200;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

// Saved transcripts are untrusted input too. Keep this data-only schema and its
// limits aligned with desktop/src/chat/chart.rs; never accept Chart.js options.
export function isChartSpec(value: unknown): value is ChartSpec {
  if (!objectWithKeys(value, ["type", "title", "height", "labels", "datasets", "xLabel", "yLabel"])) return false;
  if (!(CHART_TYPES as readonly unknown[]).includes(value.type) || !text(value.title)) return false;
  if (value.height !== undefined && !(CHART_HEIGHTS as readonly unknown[]).includes(value.height)) return false;
  if ((value.xLabel !== undefined && !text(value.xLabel)) || (value.yLabel !== undefined && !text(value.yLabel)))
    return false;
  const xy = value.type === "scatter" || value.type === "bubble";
  if (xy) {
    if (value.labels !== undefined) return false;
  } else if (
    !Array.isArray(value.labels) ||
    value.labels.length < 1 ||
    value.labels.length > 500 ||
    !value.labels.every(text)
  ) {
    return false;
  }
  if (!Array.isArray(value.datasets) || value.datasets.length < 1 || value.datasets.length > 8) return false;
  let count = 0;
  const nonnegative = value.type === "pie" || value.type === "doughnut" || value.type === "polarArea";
  for (const dataset of value.datasets) {
    if (!objectWithKeys(dataset, ["label", "data"]) || !text(dataset.label)) return false;
    if (!Array.isArray(dataset.data) || dataset.data.length < 1 || dataset.data.length > 500) return false;
    count += dataset.data.length;
    if (count > 2_000) return false;
    if (!xy && dataset.data.length !== (value.labels as string[]).length) return false;
    for (const point of dataset.data) {
      if (!xy) {
        if (!finite(point) || (nonnegative && point < 0)) return false;
      } else {
        if (!objectWithKeys(point, ["x", "y", "r"]) || !finite(point.x) || !finite(point.y)) return false;
        if (value.type === "bubble") {
          if (!finite(point.r) || point.r < 0 || point.r > 100) return false;
        } else if (point.r !== undefined) return false;
      }
    }
  }
  return true;
}
