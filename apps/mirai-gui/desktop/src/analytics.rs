use std::{sync::Mutex, time::Duration};

use serde::Serialize;
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use uzu::{
    device::Device,
    engine::Engine,
    session::api::{Client, Endpoint, RetryConfig},
    types::session::chat::{ChatReplyJoulesPerToken, ChatReplyStats},
};

use crate::error::AppResult;

const BASE_URL: &str = "https://sdk.trymirai.com/api/v2";

#[derive(Serialize)]
#[serde(tag = "event_name", content = "payload", rename_all = "snake_case")]
pub enum Event {
    #[serde(rename = "model_download_started")]
    DownloadStarted {
        model_id: String,
    },
    #[serde(rename = "model_download_finished")]
    DownloadFinished {
        model_id: String,
    },
    #[serde(rename = "model_inference_started")]
    InferenceStarted {
        model_id: String,
    },
    #[serde(rename = "model_inference_finished")]
    InferenceFinished {
        model_id: String,
        stats: Box<Stats>,
    },
    // Do not serialize engine errors: they can contain user text or local paths.
    #[serde(rename = "model_inference_failed")]
    InferenceFailed {
        error: &'static str,
    },
}

#[derive(Serialize)]
pub struct Stats {
    #[serde(flatten)]
    stats: ChatReplyStats,
    input_joules_per_token: Option<ChatReplyJoulesPerToken>,
    output_joules_per_token: Option<ChatReplyJoulesPerToken>,
}

impl From<ChatReplyStats> for Stats {
    fn from(stats: ChatReplyStats) -> Self {
        Self {
            input_joules_per_token: stats.input_joules_per_token(),
            output_joules_per_token: stats.output_joules_per_token(),
            stats,
        }
    }
}

// Explicit fields keep local paths and other Device fields off the wire.
#[derive(Serialize)]
struct DeviceInfo {
    os_name: Option<String>,
    cpu_name: Option<String>,
    memory_total: i64,
}

#[derive(Serialize)]
struct Context {
    // Retain the server's field name; the ID now belongs only to this app's
    // opt-in period and is neither persisted nor shared with the engine.
    engine_session_id: String,
    app_version: &'static str,
    uzu_version: String,
    toolchain_version: String,
    device: DeviceInfo,
}

#[derive(Serialize)]
struct Record {
    event_time: String,
    #[serde(flatten)]
    event: Event,
}

#[derive(Serialize)]
struct Body<'a> {
    #[serde(flatten)]
    context: &'a Context,
    #[serde(flatten)]
    record: &'a Record,
}

struct Events;
impl Endpoint for Events {
    const PATH: &'static str = "telemetry/events";
    type Request = Value;
    type Response = ();
}

struct Worker {
    sender: mpsc::Sender<Record>,
    task: tauri::async_runtime::JoinHandle<()>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Opting out discards queued records and cancels any pending send/retry.
        self.task.abort();
    }
}

impl Worker {
    fn start() -> Option<Self> {
        let device = Device::new().ok()?;
        let context = Context {
            engine_session_id: uuid::Uuid::new_v4().to_string(),
            app_version: env!("CARGO_PKG_VERSION"),
            uzu_version: Engine::version(),
            toolchain_version: Engine::toolchain_version(),
            device: DeviceInfo {
                os_name: device.os_name,
                cpu_name: device.cpu_name,
                memory_total: device.memory_total,
            },
        };
        let client = Client::builder()
            .base_url(BASE_URL)
            .retry(RetryConfig {
                max_attempts: 2,
                base_delay: Duration::from_secs(1),
                budget: Duration::from_secs(10),
            })
            .build()
            .ok()?;
        let (sender, mut receiver) = mpsc::channel::<Record>(64);
        let task = tauri::async_runtime::spawn(async move {
            while let Some(record) = receiver.recv().await {
                let Ok(body) = serde_json::to_value(Body {
                    context: &context,
                    record: &record,
                }) else {
                    continue;
                };
                // Reporting must never block a chat, and failures are not persisted.
                let _ = client.send::<Events>(&body).await;
            }
        });
        Some(Self {
            sender,
            task,
        })
    }
}

#[derive(Default)]
pub struct AnalyticsState {
    worker: Mutex<Option<Worker>>,
}

fn enabled(settings: &Map<String, Value>) -> bool {
    settings.get("analyticsEnabled").and_then(Value::as_bool) == Some(true)
}

fn apply(
    worker: &mut Option<Worker>,
    enabled: bool,
) {
    if !enabled {
        *worker = None;
    } else if worker.is_none() {
        *worker = Worker::start();
    }
}

impl AnalyticsState {
    pub fn restore(&self) {
        let mut worker = self.worker.lock().expect("analytics mutex poisoned");
        let settings = crate::storage::settings_load().ok();
        apply(&mut worker, settings.as_ref().and_then(Value::as_object).is_some_and(enabled));
    }

    pub fn patch_settings(
        &self,
        patch: Map<String, Value>,
    ) -> AppResult<()> {
        // Serialize persistence and application so rapid toggles cannot apply
        // out of order or disagree with the preference read on the next launch.
        let mut worker = self.worker.lock().expect("analytics mutex poisoned");
        let settings = crate::storage::settings_merge(patch)?;
        apply(&mut worker, enabled(&settings));
        Ok(())
    }

    pub fn report(
        &self,
        event: impl FnOnce() -> Event,
    ) {
        if let Some(worker) = self.worker.lock().expect("analytics mutex poisoned").as_ref() {
            let _ = worker.sender.try_send(Record {
                event_time: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                event: event(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reporting_requires_explicit_opt_in() {
        for settings in [
            serde_json::json!({}),
            serde_json::json!({"analyticsEnabled": false}),
            serde_json::json!({"analyticsEnabled": "true"}),
        ] {
            assert!(!enabled(settings.as_object().unwrap()));
        }
        assert!(enabled(serde_json::json!({"analyticsEnabled": true}).as_object().unwrap()));
        AnalyticsState::default().report(|| panic!("must not collect events without consent"));
    }

    #[tokio::test]
    async fn opting_out_aborts_worker_and_drops_queued_records() {
        let (sender, receiver) = mpsc::channel(4);
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tauri::async_runtime::spawn(async move {
            let _receiver = receiver;
            let _ = started.send(());
            std::future::pending::<()>().await;
        });
        let state = AnalyticsState {
            worker: Mutex::new(Some(Worker {
                sender: sender.clone(),
                task,
            })),
        };
        ready.await.unwrap();
        state.report(|| Event::DownloadStarted {
            model_id: "model".into(),
        });
        assert_eq!(sender.capacity(), 3);
        apply(&mut state.worker.lock().unwrap(), false);
        tokio::time::timeout(Duration::from_secs(1), sender.closed()).await.unwrap();
        state.report(|| panic!("must stop collection immediately after opting out"));
    }

    #[test]
    fn payload_contains_only_declared_metadata_and_numeric_stats() {
        let context = Context {
            engine_session_id: "ephemeral-session".into(),
            app_version: "0.6.2",
            uzu_version: "0.6.2".into(),
            toolchain_version: "0.1".into(),
            device: DeviceInfo {
                os_name: Some("macOS".into()),
                cpu_name: Some("Apple".into()),
                memory_total: 64,
            },
        };
        let record = Record {
            event_time: "2026-10-09T00:00:00Z".into(),
            event: Event::InferenceFinished {
                model_id: "model".into(),
                stats: Box::new(ChatReplyStats::default().into()),
            },
        };
        let body = serde_json::to_value(Body {
            context: &context,
            record: &record,
        })
        .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "engine_session_id": "ephemeral-session",
                "app_version": "0.6.2",
                "uzu_version": "0.6.2",
                "toolchain_version": "0.1",
                "device": {"os_name": "macOS", "cpu_name": "Apple", "memory_total": 64},
                "event_time": "2026-10-09T00:00:00Z",
                "event_name": "model_inference_finished",
                "payload": {
                    "model_id": "model",
                    "stats": {
                        "duration": 0.0,
                        "time_to_first_token": null,
                        "prefill_tokens_per_second": null,
                        "generate_tokens_per_second": null,
                        "tokens_count_input": null,
                        "tokens_count_input_cached": null,
                        "tokens_count_output": null,
                        "memory_used_bytes": null,
                        "speculator_stats": null,
                        "input_energy": null,
                        "output_energy": null,
                        "input_joules_per_token": null,
                        "output_joules_per_token": null
                    }
                }
            })
        );
    }
}
