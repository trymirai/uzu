use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

pub struct StubPowerRecorder {}

impl StubPowerRecorder {
    pub fn new() -> Self {
        Self {}
    }
}

impl PowerRecorder for StubPowerRecorder {
    fn stop(&self) -> Option<ChatReplyPowerStats> {
        None
    }
}
