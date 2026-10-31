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
  /** The model loaded in the backend, which outlives a page reload; waits out a load or eject in progress. */
  getResidentSession(): Promise<string | null>;
  ejectSession(params: { repoId: string }): Promise<void>;
};
