use std::{future::Future, time::Duration};

use tokio::time::{sleep as tokio_sleep, timeout as tokio_timeout};

pub(super) fn parse_range(value: &str) -> Option<(u64, Option<u64>)> {
    let bytes_range = value.strip_prefix("bytes=")?;
    let (start, end) = bytes_range.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    let end = if end.is_empty() {
        None
    } else {
        Some(end.parse::<u64>().ok()?)
    };
    Some((start, end))
}

pub(super) fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        206 => "Partial Content",
        302 => "Found",
        404 => "Not Found",
        416 => "Range Not Satisfiable",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Mock Status",
    }
}

pub(super) async fn wait_until<F, Fut>(
    timeout_duration: Duration,
    mut predicate: F,
) where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let result = tokio_timeout(timeout_duration, async {
        loop {
            if predicate().await {
                return;
            }
            tokio_sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(result.is_ok(), "mock server wait timed out after {:?}", timeout_duration);
}

pub(super) async fn wait_until_value<F, Fut, T>(
    timeout_duration: Duration,
    mut value: F,
) -> T
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Option<T>>,
{
    tokio_timeout(timeout_duration, async {
        loop {
            if let Some(result) = value().await {
                return result;
            }
            tokio_sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("mock server wait timed out after {:?}", timeout_duration))
}
