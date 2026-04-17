use indexmap::IndexMap;
use reqwest::Method;
use serde_json::Value;

use crate::api::Config;

pub struct Payload {
    pub query: Option<IndexMap<String, String>>,
    pub body: Option<Value>,
}

pub trait Endpoint {
    fn method(&self) -> Method;
    fn path(&self) -> String;
    fn headers(&self) -> IndexMap<String, String>;
    fn payload(
        &self,
        config: &Config,
    ) -> Payload;
}
