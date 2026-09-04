use std::{future::Future, time::Duration};

use reqwest::Response;
use tokio::time::{Instant, sleep};

use crate::api::{Error, IsTransient};

/// How hard to retry a request, and for how long in total.
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    pub max_attempts: usize,
    /// First backoff, doubled on each subsequent attempt.
    pub base_delay: Duration,
    /// Ceiling on the whole retry phase, so a caller on a startup path cannot
    /// be stalled for `max_attempts` timeouts.
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
    /// Send, retrying transient outcomes with exponential backoff until either
    /// `max_attempts` is reached or `budget` expires.
    ///
    /// `send_request` builds a fresh request on every attempt, so it is a
    /// closure rather than a future. A non-transient outcome is returned as-is,
    /// including a fatal error status: mapping the status to an [`Error`] is the
    /// caller's job. When the attempts run out the last outcome is returned, so
    /// callers see the real status instead of a synthetic timeout. Only an
    /// expired budget produces [`Error::Timeout`].
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

        for attempt in 1..=attempts {
            let result = send_request().await;
            let transient = match &result {
                Ok(response) => response.status().is_transient(),
                Err(error) => error.is_transient(),
            };
            if !transient || attempt == attempts {
                return result.map_err(Error::from);
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Error::Timeout);
            }
            sleep(delay.min(remaining)).await;
            delay = delay.saturating_mul(2);
        }

        Err(Error::Timeout)
    }
}
