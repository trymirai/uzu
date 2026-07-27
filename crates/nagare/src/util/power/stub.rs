use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

pub struct StubPowerRecorder {}

impl StubPowerRecorder {
    pub fn new() -> Self {
        Self {}
    }
}

impl PowerRecorder for StubPowerRecorder {
    fn begin(&self) {}

    fn finish(&self) -> Option<ChatReplyPowerStats> {
        None
    }
}
