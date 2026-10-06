export const runtimeSessionEjectReasons = {
  user: "user",
  auto: "auto",
} as const;

export type RuntimeSessionEjectReason = (typeof runtimeSessionEjectReasons)[keyof typeof runtimeSessionEjectReasons];

export type RuntimeSessionRef = {
  repoId: string;
};
