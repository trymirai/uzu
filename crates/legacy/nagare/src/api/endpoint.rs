use serde::{Serialize, de::DeserializeOwned};

pub trait Endpoint {
    const PATH: &'static str;

    type Request: Serialize;
    type Response: DeserializeOwned;
}
