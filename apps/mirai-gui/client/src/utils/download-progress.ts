export const toDownloadProgressKbytes = (
  completedBytes: number | null | undefined,
  totalBytes: number | null | undefined,
): { downloadedKbytes: number; totalKbytes: number } => ({
  downloadedKbytes: Math.floor((completedBytes ?? 0) / 1024),
  totalKbytes: typeof totalBytes === "number" && totalBytes > 0 ? Math.floor(totalBytes / 1024) : 0,
});
