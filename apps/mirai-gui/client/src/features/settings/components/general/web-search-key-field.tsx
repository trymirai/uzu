import { EyeIcon, EyeOffIcon } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { TextField } from "@/components/ui/text-field";
import {
  requiresWebSearchKey,
  WEB_SEARCH_PROVIDERS,
  webSearchKeyOf,
  webSearchKeyTitle,
  type WebSearchKeyProvider,
} from "@/constants/web-search-providers";
import { SettingRow } from "./setting-row";

type WebSearchKeyFieldProps = {
  provider: WebSearchKeyProvider;
  ending: string | undefined;
  onSave: (key: string) => Promise<string | null>;
  onRemove: () => void;
};

/** Owners give it `key={provider}`, so switching providers starts with an empty field. */
export function WebSearchKeyField({ provider, ending, onSave, onRemove }: WebSearchKeyFieldProps) {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [shown, setShown] = useState(false);
  const [changing, setChanging] = useState(false);
  const { label } = WEB_SEARCH_PROVIDERS[provider];
  const title = webSearchKeyTitle(provider);
  const keyPage = webSearchKeyOf(provider).page;
  const editing = !ending || changing;
  const required = requiresWebSearchKey(provider);

  const save = async () => {
    if (saving || key.trim() === "") return;
    setSaving(true);
    const failure = await onSave(key);
    setSaving(false);
    if (failure) {
      setError(failure);
      return;
    }
    setKey("");
    setShown(false);
    setChanging(false);
  };

  const cancel = () => {
    setKey("");
    setError("");
    setChanging(false);
  };

  const description = (
    <>
      {required
        ? `Required. ${label} searches only with an API key.`
        : `Optional. Add ${/^[aeiou]/i.test(label) ? "an" : "a"} ${label} API key for higher limits, or leave it empty to search without one.`}{" "}
      <a href={keyPage} target="_blank" rel="noreferrer" className="text-blue hover:underline">
        Get a key
      </a>
    </>
  );

  return (
    <SettingRow title={title} description={description}>
      <form
        className="mt-3"
        onSubmit={(event) => {
          event.preventDefault();
          void save();
        }}
      >
        {editing ? (
          <TextField
            error={error || undefined}
            controlProps={{
              type: shown ? "text" : "password",
              "aria-label": title,
              value: key,
              onChange: (event) => {
                setKey(event.target.value);
                setError("");
              },
              onKeyDown: (event) => {
                if (event.key === "Escape" && changing) cancel();
              },
              placeholder: `Paste your ${label} API key`,
              autoComplete: "off",
              spellCheck: false,
              disabled: saving,
              autoFocus: changing,
              fullWidth: true,
              rightSlot: (
                <>
                  <IconButton
                    type="button"
                    aria-label={shown ? "Hide key" : "Show key"}
                    onClick={() => setShown((value) => !value)}
                    icon={shown ? <EyeOffIcon className="size-4" /> : <EyeIcon className="size-4" />}
                    className="text-label-muted"
                  />
                  {changing && (
                    <Button kind="ghost" size="xxs" type="button" onClick={cancel} disabled={saving}>
                      Cancel
                    </Button>
                  )}
                  {key.trim() !== "" && (
                    <Button kind="secondary" size="xxs" type="submit" disabled={saving}>
                      {saving ? "Checking…" : "Save"}
                    </Button>
                  )}
                </>
              ),
            }}
          />
        ) : (
          <TextField
            controlProps={{
              "aria-label": title,
              value: `••••••••${ending}`,
              readOnly: true,
              tabIndex: -1,
              fullWidth: true,
              className: "font-mono",
              rightSlot: (
                <>
                  <Button kind="ghost" size="xxs" type="button" ariaLabel={`Remove ${title}`} onClick={onRemove}>
                    Remove
                  </Button>
                  <Button
                    kind="secondary"
                    size="xxs"
                    type="button"
                    ariaLabel={`Change ${title}`}
                    onClick={() => setChanging(true)}
                  >
                    Change
                  </Button>
                </>
              ),
            }}
          />
        )}
      </form>
    </SettingRow>
  );
}
