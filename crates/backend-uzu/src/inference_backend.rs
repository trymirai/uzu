use serde::{Deserialize, Serialize};
use shoji::traits::BackendInstance as BackendInstanceTrait;

use crate::TOOLCHAIN_VERSION;

#[derive(Debug, thiserror::Error)]
pub enum Error {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config;

pub struct Backend;

impl BackendInstanceTrait for Backend {
    type Error = Error;
    type Config = Config;

    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }

    #[allow(unused_variables)]
    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        Ok(Self)
    }
}
