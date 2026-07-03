use std::env;

use reqwest::Client;
use uzu::device::Device;

#[derive(serde::Deserialize)]
struct RecommendResponse {
    #[serde(rename = "repoId")]
    repo_id: Option<String>,
}

pub async fn fetch_repo_id() -> Option<String> {
    let base =
        env::var("SDK_API_BASE").or_else(|_| env::var("MIRAI_SDK_API_BASE")).ok()?.trim_end_matches('/').to_string();
    let mem = Device::create().ok()?.memory_total;
    let version = env::var("ENGINE_VERSION").ok().filter(|v| !v.trim().is_empty());
    let path = version.as_ref().map(|v| format!("models/recommend/{v}")).unwrap_or_else(|| "models/recommend".into());
    let url = format!("{base}/{path}");
    let client = Client::new();
    let res = client.get(url).query(&[("mem", mem)]).send().await.ok()?;
    let body: RecommendResponse = res.json().await.ok()?;
    body.repo_id.filter(|id| !id.is_empty())
}
