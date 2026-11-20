import { isWebSearchProvider, takesWebSearchKey, type WebSearchKeyProvider } from "@/constants/web-search-providers";

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

export type WebSearchStoredKey = { provider: WebSearchKeyProvider; ending: string };

export const isWebSearchStoredKey = (value: unknown): value is WebSearchStoredKey =>
  isRecord(value) &&
  isWebSearchProvider(value.provider) &&
  takesWebSearchKey(value.provider) &&
  typeof value.ending === "string";

export type WebSearchSource = { title: string; url: string };

export type WebSearchFailure =
  { kind: "limit"; provider: string } | { kind: "key"; provider: string } | { kind: "denied" } | { kind: "keychain" };

export type WebSearchItem = {
  type: "webSearch";
  query: string;
  provider?: string;
  called: boolean;
  failed?: boolean;
  failure?: WebSearchFailure;
  sources?: WebSearchSource[];
};

const isWebSearchSource = (value: unknown): value is WebSearchSource =>
  isRecord(value) && typeof value.title === "string" && typeof value.url === "string" && /^https?:\/\//.test(value.url);

const isWebSearchFailure = (value: unknown): value is WebSearchFailure =>
  isRecord(value) &&
  (value.kind === "denied" ||
    value.kind === "keychain" ||
    ((value.kind === "limit" || value.kind === "key") && typeof value.provider === "string"));

export const isWebSearchItem = (value: unknown): value is WebSearchItem =>
  isRecord(value) &&
  value.type === "webSearch" &&
  typeof value.query === "string" &&
  (value.provider === undefined || typeof value.provider === "string") &&
  typeof value.called === "boolean" &&
  (value.failed === undefined || typeof value.failed === "boolean") &&
  (value.failure === undefined || isWebSearchFailure(value.failure)) &&
  (value.sources === undefined || (Array.isArray(value.sources) && value.sources.every(isWebSearchSource)));

export type WebSearchMode = "off" | "ask" | "auto";

export const isWebSearchMode = (value: unknown): value is WebSearchMode =>
  value === "off" || value === "ask" || value === "auto";

const isStrings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

export type SearchPermissionRequest = { requestId: string; provider: string; queries: string[]; pages: string[] };

export type SearchPermissionDecision = "deny" | "allow" | "alwaysAllow";

export const toSearchPermissionRequest = (value: unknown): SearchPermissionRequest | undefined => {
  if (!isRecord(value)) return undefined;
  const { requestId, provider, queries, pages = [] } = value;
  if (typeof requestId !== "string" || typeof provider !== "string") return undefined;
  if (!isStrings(queries) || !isStrings(pages) || queries.length + pages.length === 0) return undefined;
  return { requestId, provider, queries, pages };
};

/** `url` is empty until the model has written the address. */
export type WebPageItem = {
  type: "webPage";
  url: string;
  called: boolean;
  failed?: boolean;
  source?: WebSearchSource;
};

export const isWebPageItem = (value: unknown): value is WebPageItem =>
  isRecord(value) &&
  value.type === "webPage" &&
  typeof value.url === "string" &&
  typeof value.called === "boolean" &&
  (value.failed === undefined || typeof value.failed === "boolean") &&
  (value.source === undefined || isWebSearchSource(value.source));
