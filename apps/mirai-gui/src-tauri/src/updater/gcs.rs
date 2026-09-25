use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use gcp_auth::{CustomServiceAccount, TokenProvider};

use super::UpdaterState;

const GCS_READ_ONLY: &str = "https://www.googleapis.com/auth/devstorage.read_only";

// gcp_auth caches and refreshes the token, so we fetch on demand per check.
pub(super) struct GcsAuth {
    sa: CustomServiceAccount,
}

impl GcsAuth {
    fn from_base64(base64_credentials: &str) -> Result<Self, String> {
        let bytes = STANDARD.decode(base64_credentials.trim()).map_err(|e| e.to_string())?;
        let json = String::from_utf8(bytes).map_err(|e| e.to_string())?;
        let sa = CustomServiceAccount::from_json(&json).map_err(|e| e.to_string())?;
        Ok(Self {
            sa,
        })
    }

    pub(super) async fn bearer(&self) -> Result<String, String> {
        let token = self.sa.token(&[GCS_READ_ONLY]).await.map_err(|e| e.to_string())?;
        Ok(format!("Bearer {}", token.as_str()))
    }
}

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

fn env_config() -> (String, String) {
    let bucket = config_value(std::env::var("MIRAI_CHAT_BUCKET_NAME"), option_env!("MIRAI_CHAT_BUCKET_NAME"));
    let creds = config_value(std::env::var("APP_UPDATE_CREDENTIALS"), option_env!("APP_UPDATE_CREDENTIALS"));
    (bucket, creds)
}

pub(super) async fn ensure_auth(state: &UpdaterState) -> Result<(Arc<GcsAuth>, String), String> {
    let mut inner = state.inner.lock().await;
    if inner.bucket.is_empty() {
        let (bucket, creds) = env_config();
        if bucket.is_empty() || creds.is_empty() {
            return Err("missing-update-config".to_string());
        }
        inner.auth = Some(Arc::new(GcsAuth::from_base64(&creds)?));
        inner.bucket = bucket;
    }
    let auth = inner.auth.clone().ok_or("missing-update-config")?;
    Ok((auth, inner.bucket.clone()))
}
