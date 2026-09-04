use serde::{Serialize, de::DeserializeOwned};

/// One endpoint of an API, pairing its path with the types it accepts and
/// returns so a call site cannot mismatch them.
pub trait Endpoint {
    /// Appended to the client's base URL.
    const PATH: &'static str;

    type Request: Serialize;
    type Response: DeserializeOwned;
}
