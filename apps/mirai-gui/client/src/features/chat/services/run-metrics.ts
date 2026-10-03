import type { SessionOutputStats } from "@/types/llm-stream";

export type RunMetrics = {
  ttftSec?: number;
  totalSec?: number;
  tps?: number;
  tokensOut: number;
};

export function getRunMetrics(stats: SessionOutputStats): RunMetrics {
  return {
    ttftSec: stats.prefillStats.duration,
    totalSec: stats.totalStats.duration,
    tps: stats.generateStats?.tokensPerSecond,
    tokensOut: stats.totalStats.tokensCountOutput,
  };
}
