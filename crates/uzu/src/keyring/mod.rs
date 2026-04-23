mod error;

pub use error::KeyringError;
use keyring::Entry;

const SERVICE_NAME: &str = "com.trymirai.keyring";

#[bindings::export(Class)]
pub struct Keyring {}

impl Keyring {
    pub fn new() -> Self {
        Self {}
    }
}

#[bindings::export(Implementation)]
impl Keyring {
    #[bindings::export(Method)]
    pub fn store(
        &self,
        key: String,
        value: String,
    ) -> Result<(), KeyringError> {
        let entry = Entry::new(SERVICE_NAME, &key).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })?;
        entry.set_password(&value).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })?;
        Ok(())
    }

    #[bindings::export(Method)]
    pub fn retrieve(
        &self,
        key: String,
    ) -> Result<String, KeyringError> {
        let entry = Entry::new(SERVICE_NAME, &key).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })?;
        entry.get_password().map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })
    }

    #[bindings::export(Method)]
    pub fn remove(
        &self,
        key: String,
    ) -> Result<(), KeyringError> {
        let entry = Entry::new(SERVICE_NAME, &key).map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })?;
        entry.delete_credential().map_err(|error| KeyringError::BackendError {
            message: error.to_string(),
        })?;
        Ok(())
    }
}
