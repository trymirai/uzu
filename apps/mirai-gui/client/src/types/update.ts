export const UpdateStatusPhase = {
  Idle: "idle",
  Downloading: "downloading",
  Downloaded: "downloaded",
} as const;
export type UpdateStatusPhase = (typeof UpdateStatusPhase)[keyof typeof UpdateStatusPhase];

export type UpdateStatus =
  | { phase: typeof UpdateStatusPhase.Idle }
  | { phase: typeof UpdateStatusPhase.Downloading; version: string }
  | { phase: typeof UpdateStatusPhase.Downloaded; version: string };
