import type { PlatformModel } from "@/types/models";
import { formatModelName } from "@/utils/format";

export type SizeAccessor = (model: PlatformModel) => number;

export function createSizeAccessor(getModelState: (id: string) => { totalKbytes?: number } | undefined): SizeAccessor {
  return (m: PlatformModel): number => {
    const id = m.repoId;
    const fromState = getModelState(id)?.totalKbytes ?? 0;
    const fromModel = typeof m.size === "number" ? m.size / 1000 : 0;
    return fromState > 0 ? fromState : fromModel > 0 ? fromModel : 0;
  };
}

const collator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: "base",
});

type ReleaseHint = {
  familyBase: string;
  familyVersion: number | null;
  datedSuffix: number | null;
  parameterSize: number | null;
  variantRank: number;
};

const COMMUNITY_MIRROR_OWNERS = new Set(["mlx-community"]);

const getRepoOwner = (repoId: string): string => {
  const parts = repoId.split("/");
  return (parts[0] || "").toLowerCase();
};

const normalizeFamilyBase = (value: string): string => value.replace(/[^a-z]/gi, "").toLowerCase();

const compareFamilyBase = (a: string, b: string): number => {
  if (a === b) return 0;

  if (a && b) {
    if (a.includes(b)) return 1;
    if (b.includes(a)) return -1;
  }

  return collator.compare(a, b);
};

const getRepositorySourcePenalty = (repoId: string): number => {
  const owner = getRepoOwner(repoId);
  return COMMUNITY_MIRROR_OWNERS.has(owner) ? 1 : 0;
};

const getVariantRank = (model: PlatformModel, slug: string): number => {
  const normalizedSlug = slug.toLowerCase();
  const isAwq = /(?:^|[-_])awq(?:[-_]|$)/i.test(normalizedSlug);
  const quantization = (model.quantization ?? "").toLowerCase();
  const is4Bit = quantization.includes("4bit") || /(?:^|[-_])4bit(?:[-_]|$)/i.test(normalizedSlug);
  const is8Bit = quantization.includes("8bit") || /(?:^|[-_])8bit(?:[-_]|$)/i.test(normalizedSlug);
  const isQuantized = isAwq || is4Bit || is8Bit;

  const formatRank = (() => {
    if (!isQuantized) return 0;
    if (is8Bit) return 1;
    if (is4Bit || isAwq) return 2;
    return 3;
  })();

  const sourcePenalty = getRepositorySourcePenalty(model.repoId);

  return formatRank * 10 + sourcePenalty;
};

const getReleaseHint = (model: PlatformModel): ReleaseHint => {
  const slug = formatModelName(model.repoId ?? "");

  const joinedVersionMatch = slug.match(/^([A-Za-z]+)(\d+(?:\.\d+)?)(?=[-_.]|$)/);
  if (joinedVersionMatch) {
    return {
      familyBase: normalizeFamilyBase(joinedVersionMatch[1] ?? ""),
      familyVersion: Number.parseFloat(joinedVersionMatch[2] ?? ""),
      datedSuffix: getDatedSuffix(slug),
      parameterSize: getParameterSize(slug),
      variantRank: getVariantRank(model, slug),
    };
  }

  const separatedVersionMatch = slug.match(/^([A-Za-z]+(?:-[A-Za-z]+)*)-(\d+(?:\.\d+)?)(?=[-_.]|$)/);
  if (separatedVersionMatch) {
    return {
      familyBase: normalizeFamilyBase(separatedVersionMatch[1] ?? ""),
      familyVersion: Number.parseFloat(separatedVersionMatch[2] ?? ""),
      datedSuffix: getDatedSuffix(slug),
      parameterSize: getParameterSize(slug),
      variantRank: getVariantRank(model, slug),
    };
  }

  return {
    familyBase: normalizeFamilyBase(slug.split(/[-_.]/)[0] ?? slug),
    familyVersion: null,
    datedSuffix: getDatedSuffix(slug),
    parameterSize: getParameterSize(slug),
    variantRank: getVariantRank(model, slug),
  };
};

const getDatedSuffix = (slug: string): number | null => {
  const tokens = slug.split(/[-_]/).filter(Boolean);
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const token = tokens[index];
    if (token && /^\d{4,6}$/.test(token)) {
      return Number.parseInt(token, 10);
    }
  }

  return null;
};

const getParameterSize = (slug: string): number | null => {
  const match = slug.match(/(?:^|[-_])(\d+(?:\.\d+)?)([BM])(?:[-_]|$)/i);
  if (!match) return null;

  const value = Number.parseFloat(match[1] ?? "");
  const unit = (match[2] ?? "").toUpperCase();

  if (!Number.isFinite(value)) return null;
  if (unit === "M") return value / 1000;
  return value;
};

const compareSourceIndex = (a: PlatformModel, b: PlatformModel): number => {
  const indexA = typeof a.sourceIndex === "number" ? a.sourceIndex : Number.MAX_SAFE_INTEGER;
  const indexB = typeof b.sourceIndex === "number" ? b.sourceIndex : Number.MAX_SAFE_INTEGER;
  return indexA - indexB;
};

export function createNewestTextModelComparator() {
  return (a: PlatformModel, b: PlatformModel): number => {
    const releaseA = getReleaseHint(a);
    const releaseB = getReleaseHint(b);

    const familyBaseComparison = compareFamilyBase(releaseA.familyBase, releaseB.familyBase);

    if (familyBaseComparison === 0) {
      if (
        releaseA.familyVersion !== null &&
        releaseB.familyVersion !== null &&
        releaseA.familyVersion !== releaseB.familyVersion
      ) {
        return releaseB.familyVersion - releaseA.familyVersion;
      }

      if (releaseA.familyVersion !== releaseB.familyVersion) {
        return releaseA.familyVersion === null ? 1 : -1;
      }
    } else if (familyBaseComparison !== 0) {
      return familyBaseComparison;
    }

    if (familyBaseComparison === 0 && releaseA.familyVersion === releaseB.familyVersion) {
      if (releaseA.datedSuffix !== null && releaseB.datedSuffix !== null) {
        if (releaseA.datedSuffix !== releaseB.datedSuffix) {
          return releaseB.datedSuffix - releaseA.datedSuffix;
        }
      } else if (releaseA.datedSuffix !== releaseB.datedSuffix) {
        return releaseA.datedSuffix === null ? 1 : -1;
      }
    }

    if (familyBaseComparison === 0 && releaseA.familyVersion === releaseB.familyVersion) {
      if (
        releaseA.parameterSize !== null &&
        releaseB.parameterSize !== null &&
        releaseA.parameterSize !== releaseB.parameterSize
      ) {
        return releaseB.parameterSize - releaseA.parameterSize;
      }

      if (releaseA.variantRank !== releaseB.variantRank) {
        return releaseA.variantRank - releaseB.variantRank;
      }
    }

    const bySourceIndex = compareSourceIndex(a, b);
    if (bySourceIndex !== 0) {
      return bySourceIndex;
    }

    return collator.compare(a.repoId ?? "", b.repoId ?? "");
  };
}
