#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub type SystemTime = std::time::SystemTime;

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub type SystemTime = web_time::SystemTime;
