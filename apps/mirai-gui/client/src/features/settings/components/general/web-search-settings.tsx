import { useState } from "react";
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
import { WebSearchKeyField } from "./web-search-key-field";

const PROVIDER_OPTIONS = WEB_SEARCH_PROVIDER_IDS.map((value) => ({
  value,
  label: WEB_SEARCH_PROVIDERS[value].label,
  icon: WEB_SEARCH_PROVIDERS[value].icon,
}));

type KeyEntry = {
  provider: WebSearchKeyProvider;
  open: boolean;
  opening: number;
};

export function WebSearchSettings() {
  const { webSearchProvider, setWebSearchProvider, webSearchKeys, saveWebSearchKey, removeWebSearchKey } =
    useSettingsStore();
  // Kept after the dialog closes, so its text stays while it fades out.
  const [keyEntry, setKeyEntry] = useState<KeyEntry | null>(null);
  const enterKey = (provider: WebSearchKeyProvider) =>
    setKeyEntry((previous) => ({ provider, open: true, opening: (previous?.opening ?? 0) + 1 }));

  const [removal, setRemoval] = useState<{ provider: WebSearchKeyProvider; open: boolean } | null>(null);
  const toast = useToast();

  const removeKey = async (provider: WebSearchKeyProvider) => {
    setRemoval({ provider, open: false });
    if (!(await removeWebSearchKey(provider))) toast.error("Failed to remove the key");
  };

  const selectProvider = (provider: WebSearchProvider) => {
    if (takesWebSearchKey(provider) && requiresWebSearchKey(provider) && !webSearchKeys[provider]) {
      enterKey(provider);
    } else {
      void setWebSearchProvider(provider);
    }
  };

  const saveKey = async (key: string) => {
    if (!keyEntry) return null;
    const failure = await saveWebSearchKey(keyEntry.provider, key);
    if (!failure) await setWebSearchProvider(keyEntry.provider);
    return failure;
  };

  const keyProvider = takesWebSearchKey(webSearchProvider) ? webSearchProvider : null;

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
      {keyProvider && (
        <WebSearchKeyField
          key={keyProvider}
          provider={keyProvider}
          ending={webSearchKeys[keyProvider]}
          onSave={(key) => saveWebSearchKey(keyProvider, key)}
          onRemove={() => setRemoval({ provider: keyProvider, open: true })}
        />
      )}
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
