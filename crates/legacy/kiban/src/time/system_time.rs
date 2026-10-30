#[cfg(not(target_family = "wasm"))]
pub type SystemTime = std::time::SystemTime;

#[cfg(target_family = "wasm")]
pub type SystemTime = web_time::SystemTime;
