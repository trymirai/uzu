import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { PillSelect } from "@/components/ui/select/pill-select";
import { useToast } from "@/components/ui/toast/use-toast";
import { useSettingsStore } from "@/stores/use-settings-store";
import {
  DEFAULT_WEB_SEARCH_PROVIDER,
  requiresWebSearchKey,
  takesWebSearchKey,
  webSearchKeyTitle,
  WEB_SEARCH_PROVIDER_IDS,
  WEB_SEARCH_PROVIDERS,
  type WebSearchKeyProvider,
  type WebSearchProvider,
} from "@/constants/web-search-providers";
import { SettingGroup, SettingRow } from "./setting-row";
import { WebSearchKeyDialog } from "./web-search-key-dialog";

const PROVIDER_OPTIONS = WEB_SEARCH_PROVIDER_IDS.map((value) => ({ value, label: WEB_SEARCH_PROVIDERS[value].label }));
const KEY_PROVIDERS = WEB_SEARCH_PROVIDER_IDS.filter(takesWebSearchKey).sort(
  (a, b) => Number(requiresWebSearchKey(b)) - Number(requiresWebSearchKey(a)),
);

type KeyEntry = {
  provider: WebSearchKeyProvider;
  thenSelect: boolean;
  open: boolean;
  opening: number;
};

export function WebSearchSettings() {
  const { webSearchProvider, setWebSearchProvider, webSearchKeys, saveWebSearchKey, removeWebSearchKey } =
    useSettingsStore();
  // Kept after the dialog closes, so its text stays while it fades out.
  const [keyEntry, setKeyEntry] = useState<KeyEntry | null>(null);
  const enterKey = (provider: WebSearchKeyProvider, thenSelect: boolean) =>
    setKeyEntry((previous) => ({ provider, thenSelect, open: true, opening: (previous?.opening ?? 0) + 1 }));

  const [removal, setRemoval] = useState<{ provider: WebSearchKeyProvider; open: boolean } | null>(null);
  const toast = useToast();

  const removeKey = async (provider: WebSearchKeyProvider) => {
    setRemoval({ provider, open: false });
    if (!(await removeWebSearchKey(provider))) toast.error("Failed to remove the key");
  };

  const selectProvider = (provider: WebSearchProvider) => {
    if (takesWebSearchKey(provider) && requiresWebSearchKey(provider) && !webSearchKeys[provider]) {
      enterKey(provider, true);
    } else {
      void setWebSearchProvider(provider);
    }
  };

  const saveKey = async (key: string) => {
    if (!keyEntry) return null;
    const failure = await saveWebSearchKey(keyEntry.provider, key);
    if (!failure && keyEntry.thenSelect) await setWebSearchProvider(keyEntry.provider);
    return failure;
  };

  return (
    <SettingGroup title="Web search">
      <SettingRow
        title="Search provider"
        description="Choose which service runs your web searches"
        control={
          <PillSelect
            label="Search provider"
            value={webSearchProvider}
            options={PROVIDER_OPTIONS}
            onChange={selectProvider}
          />
        }
      />
      {KEY_PROVIDERS.map((provider) => {
        const { label } = WEB_SEARCH_PROVIDERS[provider];
        const required = requiresWebSearchKey(provider);
        const ending = webSearchKeys[provider];
        const title = webSearchKeyTitle(provider);
        return (
          <SettingRow
            key={provider}
            title={title}
            description={
              ending ? (
                <span className="font-mono text-[12px]" aria-label={`Key ending in ${ending}`}>
                  ••••••••{ending}
                </span>
              ) : required ? (
                `Needed to search with ${label}`
              ) : (
                `Optional. Searches use your ${label} account`
              )
            }
            control={
              <div className="flex shrink-0 items-center gap-1.5">
                {ending && (
                  <Button
                    kind="ghost"
                    size="xs"
                    ariaLabel={`Remove ${title}`}
                    onClick={() => setRemoval({ provider, open: true })}
                  >
                    Remove
                  </Button>
                )}
                <Button
                  kind="secondary"
                  size="xs"
                  ariaLabel={`${ending ? "Change" : "Add"} ${title}`}
                  onClick={() => enterKey(provider, false)}
                >
                  {ending ? "Change" : "Add key"}
                </Button>
              </div>
            }
          />
        );
      })}
      {keyEntry && (
        <WebSearchKeyDialog
          key={keyEntry.opening}
          open={keyEntry.open}
          provider={keyEntry.provider}
          onClose={() => setKeyEntry({ ...keyEntry, open: false })}
          onSave={saveKey}
        />
      )}
      {removal && (
        <Modal
          open={removal.open}
          onClose={() => setRemoval({ ...removal, open: false })}
          title={`Remove ${webSearchKeyTitle(removal.provider)}?`}
          description={
            requiresWebSearchKey(removal.provider) && webSearchProvider === removal.provider
              ? `The key will be deleted from the macOS Keychain. Web search will switch to ${WEB_SEARCH_PROVIDERS[DEFAULT_WEB_SEARCH_PROVIDER].label}.`
              : "The key will be deleted from the macOS Keychain."
          }
          primaryLabel="Remove"
          primaryKind="danger"
          onPrimary={() => void removeKey(removal.provider)}
        />
      )}
    </SettingGroup>
  );
}
