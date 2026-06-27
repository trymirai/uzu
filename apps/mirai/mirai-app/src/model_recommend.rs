#[derive(serde::Deserialize)]
struct RecommendResponse {
    #[serde(rename = "repoId")]
    repo_id: Option<String>,
}

pub async fn fetch_repo_id() -> Option<String> {
    let base = std::env::var("SDK_API_BASE")
        .or_else(|_| std::env::var("MIRAI_SDK_API_BASE"))
        .ok()?
        .trim_end_matches('/')
        .to_string();
    let mem = uzu::device::Device::create().ok()?.memory_total;
    let version = std::env::var("ENGINE_VERSION").ok().filter(|v| !v.trim().is_empty());
    let path = version.as_ref().map(|v| format!("models/recommend/{v}")).unwrap_or_else(|| "models/recommend".into());
    let url = format!("{base}/{path}");
    let client = reqwest::Client::new();
    let res = client.get(url).query(&[("mem", mem)]).send().await.ok()?;
    let body: RecommendResponse = res.json().await.ok()?;
    body.repo_id.filter(|id| !id.is_empty())
}
