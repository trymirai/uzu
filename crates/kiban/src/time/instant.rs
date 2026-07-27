#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub type Instant = std::time::Instant;

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub type Instant = web_time::Instant;
