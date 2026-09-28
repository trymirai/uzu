import type { ReasoningSupport } from "./sampling";

export const modelDownloadPhases = {
  notDownloaded: "NotDownloaded",
  downloading: "Downloading",
  paused: "Paused",
  downloaded: "Downloaded",
  locked: "Locked",
  error: "Error",
} as const;

export type ModelDownloadPhase = (typeof modelDownloadPhases)[keyof typeof modelDownloadPhases];

export type ModelDownloadState = {
  totalKbytes: number;
  downloadedKbytes: number;
  phase: ModelDownloadPhase;
  error?: string;
  // Backend event sequence this state reflects; older events and snapshots are ignored.
  seq: number;
};

export type VendorIcons = { light: string; dark: string };

export type EngineModel = {
  identifier: string;
  repoId?: string;
  vendor: string;
  vendorIcons?: VendorIcons;
  name: string;
  familyIdentifier?: string;
  familyName?: string;
  paramSize?: number;
  isThinking: boolean;
  reasoning: ReasoningSupport;
  quantization?: string | null;
  quantizationBits?: number;
  state: ModelDownloadState;
};
