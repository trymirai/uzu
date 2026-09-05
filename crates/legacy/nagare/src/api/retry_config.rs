use std::{future::Future, time::Duration};

use reqwest::{Response, header::HeaderMap};
use tokio::time::{Instant, sleep, timeout};

use crate::api::{Error, IsTransient};

#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub base_delay: Duration,
    pub budget: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_millis(100),
            budget: Duration::from_secs(15),
        }
    }
}

impl RetryConfig {
    pub async fn send<F, Fut>(
        &self,
        mut send_request: F,
    ) -> Result<Response, Error>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<Response, reqwest::Error>>,
    {
        let attempts = self.max_attempts.max(1);
        let deadline = Instant::now() + self.budget;
        let mut delay = self.base_delay;
        let mut attempt = 0;

        loop {
            attempt += 1;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::Timeout);
            }
            let Ok(result) = timeout(remaining, send_request()).await else {
                return Err(Error::Timeout);
            };

            let transient = match &result {
                Ok(response) => response.status().is_transient(),
                Err(error) => error.is_transient(),
            };
            if !transient || attempt == attempts {
                return result.map_err(Error::from);
            }

            let requested = result.as_ref().ok().and_then(|response| retry_after(response.headers()));
            sleep(requested.unwrap_or(delay).min(deadline.saturating_duration_since(Instant::now()))).await;
            delay = delay.saturating_mul(2);
        }
    }
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let seconds = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?.trim().parse().ok()?;
    Some(Duration::from_secs(seconds))
}
