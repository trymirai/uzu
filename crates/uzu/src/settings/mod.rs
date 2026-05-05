mod error;

use std::sync::Arc;

pub use error::SettingsError;
use indexmap::IndexMap;
use keyring_core::{CredentialStore, Entry, Error as KeyringError, mock, set_default_store};
use serde::{Deserialize, Serialize};

const CONFIG_NAME: &str = "settings";
const BASE_KEY: &str = "settings";

#[derive(Default, Debug, Serialize, Deserialize)]
struct SettingsConfig {
    settings: IndexMap<String, String>,
}

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingKind {
    Config,
    Secret,
}

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Settings {
    application_identifier: String,
}

impl Settings {
    pub fn new(application_identifier: String) -> Result<Self, SettingsError> {
        let keyring_store = create_keyring_store()?;
        set_default_store(keyring_store);
        Ok(Self {
            application_identifier,
        })
    }

    fn keyring_entry(&self) -> Result<Entry, SettingsError> {
        Entry::new(&self.application_identifier, BASE_KEY).map_err(|error| SettingsError::BackendError {
            message: error.to_string(),
        })
    }

    fn keyring_load_or_create(
        &self,
        entry: &Entry,
    ) -> Result<IndexMap<String, String>, SettingsError> {
        match entry.get_password() {
            Ok(value) => {
                let settings = serde_json::from_str(&value).map_err(|error| SettingsError::BackendError {
                    message: error.to_string(),
                })?;
                Ok(settings)
            },
            Err(KeyringError::NoEntry) => Ok(IndexMap::new()),
            Err(error) => Err(SettingsError::BackendError {
                message: error.to_string(),
            }),
        }
    }

    fn config_entry(&self) -> Result<SettingsConfig, SettingsError> {
        confy::load(&self.application_identifier, CONFIG_NAME).map_err(|error| SettingsError::BackendError {
            message: error.to_string(),
        })
    }
}

#[bindings::export(Implementation)]
impl Settings {
    #[bindings::export(Method)]
    pub fn save(
        &self,
        kind: SettingKind,
        key: String,
        value: Option<String>,
    ) -> Result<(), SettingsError> {
        match kind {
            SettingKind::Config => {
                let mut config = self.config_entry()?;
                if let Some(value) = value {
                    config.settings.insert(key, value);
                } else {
                    config.settings.shift_remove(&key);
                }
                confy::store(&self.application_identifier, CONFIG_NAME, config).map_err(|error| {
                    SettingsError::BackendError {
                        message: error.to_string(),
                    }
                })?;
            },
            SettingKind::Secret => {
                let entry = self.keyring_entry()?;

                let mut settings = self.keyring_load_or_create(&entry)?;
                if let Some(value) = value {
                    settings.insert(key, value);
                } else {
                    settings.shift_remove(&key);
                }

                let value = serde_json::to_string(&settings).map_err(|error| SettingsError::BackendError {
                    message: error.to_string(),
                })?;
                entry.set_password(&value).map_err(|error| SettingsError::BackendError {
                    message: error.to_string(),
                })?;
            },
        }
        Ok(())
    }

    #[bindings::export(Method)]
    pub fn load(
        &self,
        kind: SettingKind,
        key: String,
    ) -> Result<Option<String>, SettingsError> {
        match kind {
            SettingKind::Config => {
                let config = self.config_entry()?;
                Ok(config.settings.get(&key).cloned())
            },
            SettingKind::Secret => {
                let entry = self.keyring_entry()?;
                let settings = self.keyring_load_or_create(&entry)?;
                Ok(settings.get(&key).cloned())
            },
        }
    }

    #[bindings::export(Method)]
    pub fn clear(&self) -> Result<(), SettingsError> {
        if let Ok(path) = confy::get_configuration_file_path(&self.application_identifier, CONFIG_NAME) {
            std::fs::remove_file(path).map_err(|error| SettingsError::BackendError {
                message: error.to_string(),
            })?;
        }

        let entry = self.keyring_entry()?;
        entry.delete_credential().map_err(|error| SettingsError::BackendError {
            message: error.to_string(),
        })?;

        Ok(())
    }
}

#[allow(unreachable_code)]
fn create_keyring_store() -> Result<Arc<CredentialStore>, SettingsError> {
    #[cfg(target_os = "macos")]
    {
        use objc2_foundation::NSBundle;
        if NSBundle::mainBundle().bundleIdentifier().is_some() {
            use apple_native_keyring_store::protected::Store;
            return Ok(Store::new().map_err(|error| SettingsError::BackendError {
                message: error.to_string(),
            })?);
        } else {
            use apple_native_keyring_store::keychain::Store;
            return Ok(Store::new().map_err(|error| SettingsError::BackendError {
                message: error.to_string(),
            })?);
        }
    }

    #[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
    {
        use apple_native_keyring_store::protected::Store;
        return Ok(Store::new().map_err(|error| SettingsError::BackendError {
            message: error.to_string(),
        })?);
    }

    let mock_store = mock::Store::new().map_err(|error| SettingsError::BackendError {
        message: error.to_string(),
    })?;
    Ok(mock_store)
}
