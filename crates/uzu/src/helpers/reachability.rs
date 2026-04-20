use std::time::Duration;

use reqwest::Url;
use tokio::{net::TcpStream, time::timeout};

pub async fn is_endpoint_reachable(endpoint: &str) -> bool {
    let Some(url) = Url::parse(endpoint).ok() else {
        return false;
    };
    let Some(host) = url.host_str().map(|host| host.to_string()) else {
        return false;
    };
    let Some(port) = url.port_or_known_default().map(|port| port) else {
        return false;
    };

    timeout(Duration::from_millis(200), TcpStream::connect((host, port)))
        .await
        .map(|result| result.is_ok())
        .unwrap_or(false)
}
