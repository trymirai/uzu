#[cfg(not(target_family = "wasm"))]
pub type Instant = std::time::Instant;

#[cfg(target_family = "wasm")]
pub type Instant = web_time::Instant;
