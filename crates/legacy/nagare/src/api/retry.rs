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
mod tests {
    use std::time::Duration;

    use reqwest::StatusCode;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{DEFAULT_BASE_DELAY, DEFAULT_RETRY_BUDGET};
    use crate::api::{Client, Error};

    #[derive(Debug, serde::Deserialize)]
    struct Body {
        ok: bool,
    }

    /// Answers `POST /thing` with `status` for as long as the test asks.
    async fn always(status: u16) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/thing"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;
        server
    }

    fn client(
        server: &MockServer,
        max_attempts: usize,
        base_delay: Duration,
        budget: Duration,
    ) -> Client {
        Client::builder()
            .base_url(server.uri())
            .max_attempts(max_attempts)
            .retry_base_delay(base_delay)
            .retry_budget(budget)
            .build()
            .expect("client should build")
    }

    async fn attempts(server: &MockServer) -> usize {
        server.received_requests().await.expect("server records requests").len()
    }

    #[tokio::test]
    async fn transient_failures_retry_until_one_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/thing"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/thing"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "ok": true })))
            .mount(&server)
            .await;

        let client = client(&server, 4, Duration::from_millis(1), DEFAULT_RETRY_BUDGET);
        let body: Body = client.post("thing", &()).await.expect("the third attempt should succeed");

        assert!(body.ok);
        assert_eq!(attempts(&server).await, 3, "two 503s should have been retried past");
    }

    #[tokio::test]
    async fn exhausted_attempts_surface_the_real_status() {
        let server = always(503).await;

        let error = client(&server, 3, Duration::from_millis(1), DEFAULT_RETRY_BUDGET)
            .post::<Body>("thing", &())
            .await
            .expect_err("every attempt fails");

        assert!(
            matches!(error, Error::Http { code, .. } if code == StatusCode::SERVICE_UNAVAILABLE),
            "expected the real 503, got {error:?}"
        );
        assert_eq!(attempts(&server).await, 3, "all three attempts should be spent");
    }

    #[tokio::test]
    async fn fatal_status_is_not_retried() {
        let server = always(400).await;

        let error = client(&server, 4, DEFAULT_BASE_DELAY, DEFAULT_RETRY_BUDGET)
            .post::<Body>("thing", &())
            .await
            .expect_err("400 is fatal");

        assert!(matches!(error, Error::Http { code, .. } if code == StatusCode::BAD_REQUEST));
        assert_eq!(attempts(&server).await, 1, "a fatal status must not be replayed");
    }

    /// Both the deadline check and the per-sleep cap must hold, so this covers a
    /// backoff shorter than the budget and one far longer than it.
    #[tokio::test]
    async fn an_expired_budget_stops_retrying() {
        for base_delay in [Duration::from_millis(50), Duration::from_secs(30)] {
            let server = always(503).await;
            let started = std::time::Instant::now();

            let error = client(&server, 100, base_delay, Duration::from_millis(200))
                .post::<Body>("thing", &())
                .await
                .expect_err("the budget runs out");

            assert!(matches!(error, Error::Timeout), "expected a timeout, got {error:?}");
            assert!(started.elapsed() < Duration::from_secs(5), "took {:?} on a 200ms budget", started.elapsed());
        }
    }
}
