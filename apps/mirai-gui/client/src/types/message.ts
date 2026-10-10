import type { OutputShape, SessionOutputStats } from "./llm-stream";
import type { NonSystemRole } from "./chat";

export type ParsedOutput = {
  response?: string;
  chainOfThought?: string;
};

export type PerfStats = {
  ttftSec?: number;
  totalSec?: number;
  tps?: number;
  tokensOut?: number;
};

export type MessageVersion = {
  id: string;
  text: string;
  modelId: string;
  modelName: string;
  timestamp: number;
  perf?: PerfStats;
  stats?: SessionOutputStats;
  attachmentIds?: string[];
  output?: OutputShape;
  error?: string;
};

export type Message = {
  id: string;
  text: string;
  sender: NonSystemRole;
  modelId?: string;
  modelName?: string;
  timestamp: number;
  versions?: MessageVersion[];
  currentVersionIndex?: number;
  perf?: PerfStats;
  stats?: SessionOutputStats;
  attachmentIds?: string[];
  output?: OutputShape;
  error?: string;
};
