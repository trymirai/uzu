import type { ChartSpec } from "@/types/chart";

export function chartTable(chart: ChartSpec): { headers: string[]; rows: string[][] } {
  if (chart.type === "scatter" || chart.type === "bubble") {
    const bubble = chart.type === "bubble";
    return {
      headers: ["Series", chart.xLabel ?? "X", chart.yLabel ?? "Y", ...(bubble ? ["Radius (px)"] : [])],
      rows: chart.datasets.flatMap((dataset) =>
        dataset.data.flatMap((point) =>
          typeof point === "number"
            ? []
            : [[dataset.label, String(point.x), String(point.y), ...(bubble ? [String(point.r)] : [])]],
        ),
      ),
    };
  }
  return {
    headers: [
      chart.xLabel ?? "Category",
      ...chart.datasets.map((dataset) => (chart.yLabel ? `${dataset.label} (${chart.yLabel})` : dataset.label)),
    ],
    rows: (chart.labels ?? []).map((label, index) => [
      label,
      ...chart.datasets.map((dataset) => String(dataset.data[index])),
    ]),
  };
}

const markdownText = (value: string): string =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/[\\`*_[\]{}|]/g, "\\$&")
    .replace(/\r\n|\r|\n/g, "<br>");

export function chartToMarkdown(chart: ChartSpec): string {
  const { headers, rows } = chartTable(chart);
  const row = (cells: string[]) => `| ${cells.map(markdownText).join(" | ")} |`;
  return [`**${markdownText(chart.title)}**`, "", row(headers), row(headers.map(() => "---")), ...rows.map(row)].join(
    "\n",
  );
}
