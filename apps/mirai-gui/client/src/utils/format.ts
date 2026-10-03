export const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  if (bytes < 1_000) return `${bytes} B`;
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  if (bytes < 1_000_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
};

export const formatModelSize = (sizeBytes?: number | null): string | undefined => {
  if (!sizeBytes || sizeBytes <= 0) return undefined;
  const sizeGb = sizeBytes / 1e9;
  if (sizeGb >= 1) return `${sizeGb.toFixed(sizeGb >= 10 ? 0 : 1)} GB`;
  return `${Math.round(sizeBytes / 1e6)} MB`;
};

export const formatModelName = (repoId: string): string => {
  if (!repoId) return "";
  return repoId.split("/").at(-1) || repoId;
};
