use super::UpdaterState;

// A packaged app inherits no shell environment, so a compile-time value is
// the fallback; a runtime variable wins when present.
fn config_value(
    runtime: Result<String, std::env::VarError>,
    compiled: Option<&'static str>,
) -> String {
    runtime
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| compiled.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()))
        .unwrap_or_default()
}

pub(super) async fn ensure_bucket(state: &UpdaterState) -> Result<String, String> {
    let mut inner = state.inner.lock().await;
    if inner.bucket.is_empty() {
        let bucket = config_value(std::env::var("MIRAI_CHAT_BUCKET_NAME"), option_env!("MIRAI_CHAT_BUCKET_NAME"));
        if bucket.is_empty() {
            return Err("missing-update-config".to_string());
        }
        inner.bucket = bucket;
    }
    Ok(inner.bucket.clone())
}
