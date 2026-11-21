type ProviderKey = {
  required: boolean;
  page: string;
  product?: string;
};

type ProviderDetails = { label: string; icon: string; key?: ProviderKey };

const PROVIDERS = {
  duckduckgo: { label: "DuckDuckGo", icon: "/icons/web-search/duckduckgo.svg" },
  exa: {
    label: "Exa",
    icon: "/icons/web-search/exa.svg",
    key: { required: false, page: "https://dashboard.exa.ai/api-keys" },
  },
  keenable: {
    label: "Keenable",
    icon: "/icons/web-search/keenable.png",
    key: { required: false, page: "https://app.keenable.ai/console/api-keys" },
  },
  brave: {
    label: "Brave",
    icon: "/icons/web-search/brave.svg",
    key: { required: true, page: "https://brave.com/search/api/", product: "Brave Search" },
  },
} as const satisfies Record<string, ProviderDetails>;

export type WebSearchProvider = keyof typeof PROVIDERS;

export type WebSearchKeyProvider = {
  [Provider in WebSearchProvider]: (typeof PROVIDERS)[Provider] extends { key: ProviderKey } ? Provider : never;
}[WebSearchProvider];

export const WEB_SEARCH_PROVIDERS: Record<WebSearchProvider, ProviderDetails> = PROVIDERS;

const PROVIDER_IDS: readonly string[] = Object.keys(PROVIDERS);

export const isWebSearchProvider = (value: unknown): value is WebSearchProvider =>
  typeof value === "string" && PROVIDER_IDS.includes(value);

export const takesWebSearchKey = (provider: WebSearchProvider): provider is WebSearchKeyProvider =>
  WEB_SEARCH_PROVIDERS[provider].key !== undefined;

export const requiresWebSearchKey = (provider: WebSearchProvider): boolean =>
  WEB_SEARCH_PROVIDERS[provider].key?.required === true;

export const webSearchKeyOf = (provider: WebSearchKeyProvider): ProviderKey => PROVIDERS[provider].key;

export const webSearchKeyTitle = (provider: WebSearchKeyProvider): string =>
  `${webSearchKeyOf(provider).product ?? PROVIDERS[provider].label} API key`;

export const WEB_SEARCH_PROVIDER_IDS = PROVIDER_IDS.filter(isWebSearchProvider);

export const DEFAULT_WEB_SEARCH_PROVIDER: WebSearchProvider = "duckduckgo";

export const toWebSearchProvider = (value: unknown): WebSearchProvider =>
  isWebSearchProvider(value) ? value : DEFAULT_WEB_SEARCH_PROVIDER;
