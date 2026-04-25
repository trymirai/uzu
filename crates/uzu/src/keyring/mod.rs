mod error;

use std::collections::HashMap;

pub use error::KeyringError;
use keyring::{Entry, Error as BackendError};

use crate::device::Device;

const SERVICE_PREFIX: &str = "com.trymirai.keyring";
const BASE_KEY: &str = "keys";

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Keyring {
    device: Device,
    service_name: String,
}

impl Keyring {
    pub fn new() -> Result<Self, KeyringError> {
        let device = Device::new()?;
        let service_name = format!("{SERVICE_PREFIX}.{}", device.application_identifier);
        Ok(Self {
            device,
            service_name,
        })
    }

    fn entry(&self) -> Result<Entry, KeyringError> {
        Entry::new(&self.service_name, BASE_KEY).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })
    }

    fn load(&self) -> Result<HashMap<String, String>, KeyringError> {
        let entry = self.entry()?;
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

    fn save(
        &self,
        entries: &HashMap<String, String>,
    ) -> Result<(), KeyringError> {
        let entry = self.entry()?;
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
        let mut entries = self.load()?;
        entries.insert(key, value);
        self.save(&entries)
    }

    #[bindings::export(Method)]
    pub fn retrieve(
        &self,
        key: String,
    ) -> Option<String> {
        if !self.device.is_keyring_available {
            return None;
        }
        self.load().ok()?.get(&key).cloned()
    }

    #[bindings::export(Method)]
    pub fn remove(
        &self,
        key: String,
    ) -> Result<(), KeyringError> {
        if !self.device.is_keyring_available {
            return Ok(());
        }
        let mut entries = self.load()?;
        if entries.remove(&key).is_some() {
            self.save(&entries)?;
        }
        Ok(())
    }
}
