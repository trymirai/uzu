export type SessionStatePayload = {
  active: boolean;
  repoId?: string;
  isEjecting?: boolean;
};

export type SessionLoadingPayload = {
  status: "start" | "error";
  repoId: string;
};

export type SessionService = {
  onSessionState(cb: (payload: SessionStatePayload) => void): () => void;
  onSessionLoading(cb: (payload: SessionLoadingPayload) => void): () => void;
  ejectSession(params: { repoId: string }): Promise<void>;
};
