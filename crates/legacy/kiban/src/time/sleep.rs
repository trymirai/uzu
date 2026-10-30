use std::time::Duration;

#[cfg(not(target_family = "wasm"))]
pub async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await
}

#[cfg(target_family = "wasm")]
pub async fn sleep(duration: Duration) {
    crate::time::sleep_wasm::sleep(duration).await
}
