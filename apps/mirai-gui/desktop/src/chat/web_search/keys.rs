use std::collections::BTreeMap;

use security_framework::passwords::{delete_generic_password, get_generic_password, set_generic_password};
use serde::{Deserialize, Serialize};

use super::{KeyedProvider, Provider, SearchError, SearchResult};

const ENTRY: Entry = Entry {
    service: "com.mirai.tech.chat.web-search",
};
const ACCOUNT: &str = "keys";
const ERR_SEC_ITEM_NOT_FOUND: i32 = -25300;
const ENDINGS_SETTING_KEY: &str = "webSearchKeyEndings";
const ENDING_CHARS: usize = 4;
const CHECK_QUERY: &str = "weather";

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredKey {
    provider: Provider,
    ending: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Keys(BTreeMap<Provider, String>);

impl Keys {
    fn parse(entry: &[u8]) -> Self {
        let stored: BTreeMap<String, String> = serde_json::from_slice(entry).unwrap_or_default();
        Self(
            stored
                .into_iter()
                .filter_map(|(id, key)| Some((serde_json::from_value::<Provider>(id.into()).ok()?, key)))
                .collect(),
        )
    }

    pub(super) fn get(
        &self,
        provider: Provider,
    ) -> Option<&str> {
        self.0.get(&provider).map(String::as_str).filter(|key| !key.is_empty())
    }

    fn endings(&self) -> Vec<StoredKey> {
        let ending = |key: &str| key.chars().skip(key.chars().count().saturating_sub(ENDING_CHARS)).collect();
        self.0
            .iter()
            .filter(|(_, key)| !key.is_empty())
            .map(|(&provider, key)| StoredKey {
                provider,
                ending: ending(key),
            })
            .collect()
    }
}

struct Entry {
    service: &'static str,
}

impl Entry {
    fn read(&self) -> Result<Keys, String> {
        match get_generic_password(self.service, ACCOUNT) {
            Ok(entry) => Ok(Keys::parse(&entry)),
            Err(error) if error.code() == ERR_SEC_ITEM_NOT_FOUND => Ok(Keys::default()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn save(
        &self,
        provider: Provider,
        key: &str,
    ) -> Result<Keys, String> {
        let mut keys = self.read()?;
        keys.0.insert(provider, key.to_string());
        self.write(&keys)?;
        Ok(keys)
    }

    fn remove(
        &self,
        provider: Provider,
    ) -> Result<Keys, String> {
        let mut keys = self.read()?;
        if keys.0.remove(&provider).is_some() {
            self.write(&keys)?;
        }
        Ok(keys)
    }

    fn write(
        &self,
        keys: &Keys,
    ) -> Result<(), String> {
        if keys.0.is_empty() {
            return match delete_generic_password(self.service, ACCOUNT) {
                Ok(()) => Ok(()),
                Err(error) if error.code() == ERR_SEC_ITEM_NOT_FOUND => Ok(()),
                Err(error) => Err(error.to_string()),
            };
        }
        let entry = serde_json::to_vec(&keys.0).map_err(|error| error.to_string())?;
        set_generic_password(self.service, ACCOUNT, &entry).map_err(|error| error.to_string())
    }
}

pub(in crate::chat) fn key_endings() -> Vec<StoredKey> {
    crate::storage::settings_load().map(|settings| key_endings_in(&settings)).unwrap_or_default()
}

pub(super) fn has_ending(provider: Provider) -> bool {
    key_endings().iter().any(|stored| stored.provider == provider)
}

fn key_endings_in(settings: &serde_json::Value) -> Vec<StoredKey> {
    settings.get(ENDINGS_SETTING_KEY).and_then(|endings| Vec::deserialize(endings).ok()).unwrap_or_default()
}

fn save_key_endings(keys: &Keys) -> Vec<StoredKey> {
    let endings = keys.endings();
    let patch = serde_json::Map::from_iter([(ENDINGS_SETTING_KEY.to_string(), serde_json::json!(endings))]);
    if let Err(error) = crate::storage::settings_merge(patch) {
        crate::logger::warn("web-search:key:unshown", Some(serde_json::json!({ "error": error.to_string() })));
    }
    endings
}

pub(super) fn read_for_search() -> Result<Keys, String> {
    let keys = ENTRY.read()?;
    if key_endings() != keys.endings() {
        save_key_endings(&keys);
    }
    Ok(keys)
}

pub(in crate::chat) async fn save_key(
    provider: &str,
    key: &str,
) -> Result<Vec<StoredKey>, String> {
    let keyed = keyed_provider(provider)?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err("Paste the key first".into());
    }
    if !key.chars().all(|character| character.is_ascii_graphic()) {
        return Err("This key has characters a key cannot contain".into());
    }
    let provider = keyed.provider();
    check_verdict(provider, keyed.search(CHECK_QUERY, Some(&key)).await)?;
    let saved = tokio::task::spawn_blocking(move || ENTRY.save(provider, &key).map(|keys| save_key_endings(&keys)));
    saved.await.map_err(|error| error.to_string())?
}

fn check_verdict(
    provider: Provider,
    outcome: Result<Vec<SearchResult>, SearchError>,
) -> Result<(), String> {
    match outcome {
        Ok(_) | Err(SearchError::RateLimited(_)) => Ok(()),
        Err(SearchError::KeyRejected(_)) => Err(format!("{} did not accept this key", provider.label())),
        Err(error) => {
            crate::logger::info(
                "web-search:key:unchecked",
                Some(serde_json::json!({ "provider": provider, "error": error.log_entry() })),
            );
            Err(format!("Could not check the key with {}. Try again.", provider.label()))
        },
    }
}

pub(in crate::chat) async fn remove_key(provider: &str) -> Result<Vec<StoredKey>, String> {
    let provider = keyed_provider(provider)?.provider();
    let removed = tokio::task::spawn_blocking(move || ENTRY.remove(provider).map(|keys| save_key_endings(&keys)));
    removed.await.map_err(|error| error.to_string())?
}

fn keyed_provider(id: &str) -> Result<KeyedProvider, String> {
    serde_json::from_value::<Provider>(id.into())
        .ok()
        .and_then(Provider::keyed)
        .ok_or_else(|| format!("{id} takes no API key"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "writes to the Keychain"]
    fn live_keys_share_one_entry_that_goes_away_with_the_last_key() {
        const TEST_ENTRY: Entry = Entry {
            service: "com.mirai.tech.chat.web-search.test",
        };
        let _ = delete_generic_password(TEST_ENTRY.service, ACCOUNT);
        assert_eq!(TEST_ENTRY.read(), Ok(Keys::default()));

        TEST_ENTRY.save(Provider::Keenable, "first-test-value").unwrap();
        TEST_ENTRY.save(Provider::Keenable, "second-test-value").unwrap();
        let keys = TEST_ENTRY.save(Provider::Brave, "brave-test-value").unwrap();
        assert_eq!(TEST_ENTRY.read().as_ref(), Ok(&keys));
        assert_eq!(keys.get(Provider::Keenable), Some("second-test-value"));
        assert_eq!(keys.get(Provider::Brave), Some("brave-test-value"));
        assert_eq!(keys.get(Provider::Exa), None);

        let keys = TEST_ENTRY.remove(Provider::Keenable).unwrap();
        assert_eq!(keys.get(Provider::Keenable), None);
        assert_eq!(TEST_ENTRY.read().unwrap().get(Provider::Brave), Some("brave-test-value"));

        TEST_ENTRY.remove(Provider::Brave).unwrap();
        TEST_ENTRY.remove(Provider::Brave).unwrap();
        assert_eq!(
            get_generic_password(TEST_ENTRY.service, ACCOUNT).map_err(|error| error.code()),
            Err(ERR_SEC_ITEM_NOT_FOUND),
            "the entry is gone with its last key"
        );
    }

    #[test]
    fn settings_get_the_end_of_a_key_and_never_the_key() {
        let keys = Keys(BTreeMap::from([
            (Provider::Brave, "BSA9f2c0123456789a1b2".to_string()),
            (Provider::Exa, "ab".to_string()),
            (Provider::Keenable, String::new()),
        ]));
        assert_eq!(
            serde_json::to_value(keys.endings()).unwrap(),
            serde_json::json!([{ "provider": "exa", "ending": "ab" }, { "provider": "brave", "ending": "a1b2" }])
        );
    }

    #[test]
    fn the_keychain_entry_names_each_provider_as_the_settings_do() {
        let keys = Keys(BTreeMap::from([(Provider::Brave, "one".to_string())]));
        let entry = serde_json::to_string(&keys.0).unwrap();
        assert_eq!(entry, r#"{"brave":"one"}"#);
        assert_eq!(Keys::parse(entry.as_bytes()), keys);

        let with_unknown = br#"{"brave":"one","nobody":"two"}"#;
        assert_eq!(Keys::parse(with_unknown), keys, "a provider this build does not know costs no other key");
        assert_eq!(Keys::parse(b"not json"), Keys::default());
    }

    #[test]
    fn key_endings_are_read_from_the_settings_file() {
        let endings =
            serde_json::json!([{ "provider": "exa", "ending": "ab" }, { "provider": "brave", "ending": "a1b2" }]);
        let settings = serde_json::json!({ ENDINGS_SETTING_KEY: endings });
        assert_eq!(serde_json::json!(key_endings_in(&settings)), endings);

        for unreadable in [
            serde_json::json!({}),
            serde_json::json!({ ENDINGS_SETTING_KEY: "a1b2" }),
            serde_json::json!({ ENDINGS_SETTING_KEY: [{ "provider": "nobody", "ending": "a1b2" }] }),
        ] {
            assert_eq!(key_endings_in(&unreadable), Vec::new());
        }
    }

    #[test]
    fn only_a_provider_that_takes_a_key_can_be_given_one() {
        assert_eq!(keyed_provider("exa"), Ok(KeyedProvider::Exa));
        assert_eq!(keyed_provider("duckduckgo"), Err("duckduckgo takes no API key".into()));
        assert_eq!(keyed_provider("nobody"), Err("nobody takes no API key".into()));
    }

    #[test]
    fn a_key_is_saved_when_the_provider_knows_it_even_with_its_quota_spent() {
        let verdict = |outcome| check_verdict(Provider::Brave, outcome);
        assert_eq!(verdict(Ok(Vec::new())), Ok(()));
        assert_eq!(verdict(Err(SearchError::RateLimited(Provider::Brave))), Ok(()));
        assert_eq!(
            verdict(Err(SearchError::KeyRejected(Provider::Brave))),
            Err("Brave did not accept this key".into())
        );
        assert_eq!(
            verdict(Err(SearchError::failed(Provider::Brave, "HTTP 502"))),
            Err("Could not check the key with Brave. Try again.".into())
        );
    }

    #[tokio::test]
    async fn a_key_that_no_header_can_carry_is_refused_before_any_request() {
        for malformed in ["abc\u{201d}def", "two words", "line\nbreak"] {
            assert_eq!(
                save_key("brave", malformed).await,
                Err("This key has characters a key cannot contain".into()),
                "{malformed:?}"
            );
        }
        assert_eq!(save_key("brave", "  ").await, Err("Paste the key first".into()));
    }
}
