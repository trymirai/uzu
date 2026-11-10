import { modelDownloadPhases } from "@/types/model-manager";
import type { PlatformModel } from "@/types/models";
import { createNewestTextModelComparator } from "./model-sort";
import type { ModelCardState } from "../components/model-card/types";

export function toModelCardState(phase?: string, downloadedKbytes?: number, totalKbytes?: number): ModelCardState {
  const progress =
    totalKbytes && totalKbytes > 0 ? Math.min(100, Math.round(((downloadedKbytes ?? 0) / totalKbytes) * 100)) : 0;

  const mappedState: Record<string, ModelCardState> = {
    [modelDownloadPhases.initializing]: { status: "initializing" },
    [modelDownloadPhases.downloaded]: { status: "downloaded" },
    [modelDownloadPhases.paused]: { status: "paused", progress },
    [modelDownloadPhases.downloading]: { status: "downloading", progress },
    [modelDownloadPhases.error]: progress > 0 ? { status: "error", progress } : { status: "error" },
    [modelDownloadPhases.notDownloaded]: { status: "available" },
    [modelDownloadPhases.locked]: { status: "available" },
  };

  return mappedState[phase ?? modelDownloadPhases.notDownloaded] ?? { status: "available" };
}

function formatParamSize(params?: number): string | undefined {
  if (!params || params <= 0) return undefined;
  const billions = params / 1e9;
  if (billions >= 1) {
    return billions >= 10 ? `${Math.round(billions)}B` : `${billions.toFixed(1).replace(/\.0$/, "")}B`;
  }
  const millions = params / 1e6;
  return `${Math.round(millions)}M`;
}

export function formatParamRange(min?: number, max?: number): string | undefined {
  if (!min || !max) return undefined;
  if (min === max) return formatParamSize(min);
  const lo = formatParamSize(min);
  const hi = formatParamSize(max);
  if (!lo || !hi) return undefined;
  return `${lo} – ${hi}`;
}

export function formatQuantization(model: { quantization?: string | null; quantizationBits?: number }): string {
  if (!model.quantization) return "Unquantized";
  // Mirai's own quantizations read as "Mirai M" / "Mirai L", like the model menus; others keep
  // their upper-case method name.
  const mirai = /^mirai-(\w+)$/i.exec(model.quantization);
  const method = mirai?.[1] ? `Mirai ${mirai[1].toUpperCase()}` : model.quantization.toUpperCase();
  return typeof model.quantizationBits === "number" ? `${method} · ${model.quantizationBits}-bit` : method;
}

export type FamilyEntry = {
  familyIdentifier: string;
  familyName: string;
  vendor: string;
  totalCount: number;
  installedCount: number;
  paramRange: { min: number; max: number } | null;
  lastActiveAt: number;
  representative: PlatformModel;
};

type FamilyAccumulator = {
  familyIdentifier: string;
  familyName: string;
  vendor: string;
  totalCount: number;
  installedCount: number;
  paramSizes: number[];
  lastActiveAt: number;
  representative: PlatformModel;
};

function familyKey(model: PlatformModel): string {
  return model.familyIdentifier ?? model.family ?? model.vendor ?? model.repoId;
}

export function buildFamiliesView(params: {
  models: PlatformModel[];
  modelPhasesById: Record<string, string | undefined>;
  installedAtById: Record<string, number>;
  searchLower: string;
}): FamilyEntry[] {
  const { models, modelPhasesById, installedAtById, searchLower } = params;

  const accByKey = new Map<string, FamilyAccumulator>();
  for (const model of models) {
    const key = familyKey(model);
    const existing = accByKey.get(key);
    const phase = modelPhasesById[model.repoId];
    const isInstalled = phase === modelDownloadPhases.downloaded;
    const stamp = installedAtById[model.repoId] ?? 0;
    if (existing) {
      existing.totalCount += 1;
      if (isInstalled) existing.installedCount += 1;
      if (typeof model.paramSize === "number") existing.paramSizes.push(model.paramSize);
      if (stamp > existing.lastActiveAt) existing.lastActiveAt = stamp;
    } else {
      accByKey.set(key, {
        familyIdentifier: key,
        familyName: model.family ?? model.name,
        vendor: model.vendor,
        totalCount: 1,
        installedCount: isInstalled ? 1 : 0,
        paramSizes: typeof model.paramSize === "number" ? [model.paramSize] : [],
        lastActiveAt: stamp,
        representative: model,
      });
    }
  }

  const entries: FamilyEntry[] = Array.from(accByKey.values()).map((acc) => ({
    familyIdentifier: acc.familyIdentifier,
    familyName: acc.familyName,
    vendor: acc.vendor,
    totalCount: acc.totalCount,
    installedCount: acc.installedCount,
    paramRange:
      acc.paramSizes.length > 0 ? { min: Math.min(...acc.paramSizes), max: Math.max(...acc.paramSizes) } : null,
    lastActiveAt: acc.lastActiveAt,
    representative: acc.representative,
  }));

  const filtered = searchLower
    ? entries.filter(
        (e) => e.familyName.toLowerCase().includes(searchLower) || e.vendor.toLowerCase().includes(searchLower),
      )
    : entries;

  if (searchLower) {
    return filtered.sort((a, b) => a.familyName.localeCompare(b.familyName));
  }

  // A fixed order: Qwen first, then LFM, then Glimmer, then anything else. Within one line, the
  // newest version first (Qwen3.8 above Qwen3.6 above Qwen3.5), whether or not it is downloaded.
  const compareModels = createNewestTextModelComparator();
  return filtered.sort(
    (a, b) =>
      familyRank(a.familyName) - familyRank(b.familyName) ||
      b.familyName.localeCompare(a.familyName, undefined, { numeric: true }) ||
      compareModels(a.representative, b.representative),
  );
}

const FAMILY_ORDER = [/^qwen/i, /^lfm/i, /glimmer/i];

function familyRank(familyName: string): number {
  const rank = FAMILY_ORDER.findIndex((pattern) => pattern.test(familyName));
  return rank === -1 ? FAMILY_ORDER.length : rank;
}

export type FamilyDetailView = {
  installed: PlatformModel[];
  available: PlatformModel[];
};

export function buildFamilyDetailView(params: {
  models: PlatformModel[];
  modelPhasesById: Record<string, string | undefined>;
  familyIdentifier: string;
  searchLower: string;
  sortModels: (list: PlatformModel[]) => PlatformModel[];
}): FamilyDetailView {
  const { models, modelPhasesById, familyIdentifier, searchLower, sortModels } = params;

  const installed: PlatformModel[] = [];
  const available: PlatformModel[] = [];

  for (const model of models) {
    if (familyKey(model) !== familyIdentifier) continue;
    if (searchLower) {
      const haystack = `${model.name} ${formatQuantization(model)}`.toLowerCase();
      if (!haystack.includes(searchLower)) continue;
    }
    // Only a finished download is usable, so only it moves up to Downloaded; one still downloading,
    // paused or failed stays in Available, where its row shows the progress.
    if (modelPhasesById[model.repoId] === modelDownloadPhases.downloaded) installed.push(model);
    else available.push(model);
  }

  return {
    installed: sortModels(installed),
    available: sortModels(available),
  };
}
