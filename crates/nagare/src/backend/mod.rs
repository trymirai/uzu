use std::{error::Error, fmt::Debug};

use serde::{Serialize, de::DeserializeOwned};

pub trait Backend: Debug + Copy + 'static {
    type Error: Error;
    type Config: Serialize + DeserializeOwned;
    type BackendInstance: BackendInstance<Backend = Self>;
}

pub trait BackendInstance: Sized {
    type Backend: Backend<BackendInstance = Self>;

    fn new(config: <Self::Backend as Backend>::Config) -> Result<Self, <Self::Backend as Backend>::Error>;
}
