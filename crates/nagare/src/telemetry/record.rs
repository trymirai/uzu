use super::TelemetryEvent;

pub(super) struct TelemetryRecord {
    pub(super) event_time: String,
    pub(super) event: TelemetryEvent,
}

impl TelemetryRecord {
    pub(super) fn new(event: TelemetryEvent) -> Self {
        Self {
            event_time: now_rfc3339(),
            event,
        }
    }
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::TelemetryRecord;
    use crate::telemetry::TelemetryEvent;

    #[test]
    fn new_stamps_rfc3339_utc() {
        let record = TelemetryRecord::new(TelemetryEvent::ModelDownloadFinished {
            model_id: "model-1".to_string(),
        });
        assert!(record.event_time.ends_with('Z'), "expected Z suffix: {}", record.event_time);
        assert!(chrono::DateTime::parse_from_rfc3339(&record.event_time).is_ok());
    }
}
