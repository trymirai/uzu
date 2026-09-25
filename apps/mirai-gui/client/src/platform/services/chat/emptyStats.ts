import type { SessionOutputStats } from "@/utils/llm-stream";

export const emptyStats = (): SessionOutputStats => ({
  prefillStats: {
    duration: 0,
    tokensCount: 0,
    tokensPerSecond: 0,
  },
  totalStats: { duration: 0, tokensCountInput: 0, tokensCountOutput: 0 },
});
