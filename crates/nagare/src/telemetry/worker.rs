use serde_json::Value;
use tokio::sync::mpsc;

use super::{endpoint::TelemetryEndpoint, record::TelemetryRecord};
use crate::api::{Client, Error};

const MAX_RETRIES: u32 = 4;
const BASE_BACKOFF_MS: u64 = 1_000;
const MAX_BACKOFF_MS: u64 = 16_000;

pub(super) async fn run(
    client: Client,
    path: String,
    context: Value,
    mut receiver: mpsc::Receiver<TelemetryRecord>,
) {
    while let Some(record) = receiver.recv().await {
        send_with_retry(&client, &path, &context, &record).await;
    }
}

#[derive(Debug, PartialEq)]
enum Disposition {
    Retry,
    Drop,
}

async fn send_with_retry(
    client: &Client,
    path: &str,
    context: &Value,
    record: &TelemetryRecord,
) {
    let endpoint = TelemetryEndpoint::new(path.to_string(), context, record);
    let mut attempt: u32 = 0;
    loop {
        match client.send(&endpoint).await {
            Ok(()) => return,
            Err(error) => match classify(&error) {
                Disposition::Drop => {
                    tracing::warn!(%error, "telemetry event dropped");
                    return;
                },
                Disposition::Retry => {
                    if attempt >= MAX_RETRIES {
                        tracing::warn!(%error, attempts = attempt, "telemetry event dropped after retries");
                        return;
                    }
                    tokio::time::sleep(backoff(attempt)).await;
                    attempt += 1;
                },
            },
        }
    }
}

fn classify(error: &Error) -> Disposition {
    match error {
        Error::Http {
            code,
            ..
        } if *code >= 500 => Disposition::Retry,
        Error::Http {
            ..
        } => Disposition::Drop,
        Error::Timeout | Error::Network(_) => Disposition::Retry,
        Error::Decode(_) => Disposition::Drop,
    }
}

fn backoff(attempt: u32) -> std::time::Duration {
    let millis = BASE_BACKOFF_MS.saturating_mul(2u64.saturating_pow(attempt)).min(MAX_BACKOFF_MS);
    std::time::Duration::from_millis(millis)
}

#[cfg(test)]
mod tests {
    use super::{Disposition, classify};
    use crate::api::Error;

    fn http(code: u16) -> Error {
        Error::Http {
            code,
            body: String::new(),
        }
    }

    #[test]
    fn classify_400_drops_503_retries() {
        assert_eq!(classify(&http(400)), Disposition::Drop);
        assert_eq!(classify(&http(404)), Disposition::Drop);
        assert_eq!(classify(&http(503)), Disposition::Retry);
        assert_eq!(classify(&http(500)), Disposition::Retry);
        assert_eq!(classify(&http(502)), Disposition::Retry);
        assert_eq!(classify(&Error::Timeout), Disposition::Retry);
        assert_eq!(classify(&Error::Network("x".to_string())), Disposition::Retry);
        assert_eq!(classify(&Error::Decode("x".to_string())), Disposition::Drop);
    }
}
