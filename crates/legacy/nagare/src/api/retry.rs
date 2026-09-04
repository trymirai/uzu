use std::{future::Future, time::Duration};

use reqwest::Response;
use tokio::time::{Instant, sleep};

use crate::api::{Error, IsTransient};

pub const DEFAULT_MAX_ATTEMPTS: usize = 4;
pub const DEFAULT_RETRY_BUDGET: Duration = Duration::from_secs(15);
pub const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(100);

/// Send, retrying transient outcomes with exponential backoff until either
/// `max_attempts` is reached or `budget` expires.
///
/// `send_request` builds a fresh request on every attempt, so it is a closure
/// rather than a future. A non-transient outcome is returned as-is, including a
/// fatal error status: mapping the status to an [`Error`] is the caller's job.
/// When the attempts run out the last outcome is returned, so callers see the
/// real status instead of a synthetic timeout. Only an expired budget produces
/// [`Error::Timeout`].
pub async fn send<F, Fut>(
    max_attempts: usize,
    base_delay: Duration,
    budget: Duration,
    mut send_request: F,
) -> Result<Response, Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Response, reqwest::Error>>,
{
    let deadline = Instant::now() + budget;
    let mut delay = base_delay;

    for attempt in 1..=max_attempts.max(1) {
        let result = send_request().await;
        let transient = match &result {
            Ok(response) => response.status().is_transient(),
            Err(error) => error.is_transient(),
        };
        if !transient || attempt == max_attempts.max(1) {
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

#[cfg(test)]
#[path = "../../tests/unit/api/retry_test.rs"]
mod tests;
