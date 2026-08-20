use std::time::Duration;

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub async fn sleep(duration: Duration) {
    crate::time::sleep_wasm::sleep(duration).await
}
