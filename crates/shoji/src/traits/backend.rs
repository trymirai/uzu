use std::error::Error;

use serde::{Serialize, de::DeserializeOwned};

pub trait BackendInstance: Send + Sync {
    type Error: Error;
    type Config: Serialize + DeserializeOwned;

    fn identifier(&self) -> String;
    fn version(&self) -> String;

    fn new(config: Self::Config) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

pub trait Backend: Send + Sync {
    fn identifier(&self) -> String;
    fn version(&self) -> String;
}

impl<B: BackendInstance> Backend for B {
    fn identifier(&self) -> String {
        self.identifier()
    }

    fn version(&self) -> String {
        self.version()
    }
}
