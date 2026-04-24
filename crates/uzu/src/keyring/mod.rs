mod error;

use std::collections::HashMap;

pub use error::KeyringError;
use keyring::{Entry, Error as BackendError};

use crate::device::Device;

const SERVICE_NAME: &str = "com.trymirai.keyring";
const BASE_KEY: &str = "keys";

#[bindings::export(Class)]
pub struct Keyring {
    device: Device,
}

impl Keyring {
    pub fn new() -> Result<Self, KeyringError> {
        Ok(Self {
            device: Device::new()?,
        })
    }

    fn entry() -> Result<Entry, KeyringError> {
        Entry::new(SERVICE_NAME, BASE_KEY).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })
    }

    fn load() -> Result<HashMap<String, String>, KeyringError> {
        let entry = Self::entry()?;
        match entry.get_password() {
            Ok(value) => serde_json::from_str(&value).map_err(|error| KeyringError::BackendError {
                message: error.to_string(),
            }),
            Err(BackendError::NoEntry) => Ok(HashMap::new()),
            Err(error) => Err(KeyringError::BackendError {
                message: error.to_string(),
            }),
        }
    }

    fn save(entries: &HashMap<String, String>) -> Result<(), KeyringError> {
        let entry = Self::entry()?;
        let value = serde_json::to_string(entries).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })?;
        entry.set_password(&value).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })
    }
}

#[bindings::export(Implementation)]
impl Keyring {
    #[bindings::export(Factory)]
    pub fn create() -> Result<Self, KeyringError> {
        Self::new()
    }

    #[bindings::export(Method)]
    pub fn store(
        &self,
        key: String,
        value: String,
    ) -> Result<(), KeyringError> {
        if !self.device.is_keyring_available {
            return Ok(());
        }
        let mut entries = Self::load()?;
        entries.insert(key, value);
        Self::save(&entries)
    }

    #[bindings::export(Method)]
    pub fn retrieve(
        &self,
        key: String,
    ) -> Option<String> {
        if !self.device.is_keyring_available {
            return None;
        }
        Self::load().ok()?.get(&key).cloned()
    }

    #[bindings::export(Method)]
    pub fn remove(
        &self,
        key: String,
    ) -> Result<(), KeyringError> {
        if !self.device.is_keyring_available {
            return Ok(());
        }
        let mut entries = Self::load()?;
        if entries.remove(&key).is_some() {
            Self::save(&entries)?;
        }
        Ok(())
    }
}
