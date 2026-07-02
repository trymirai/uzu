#![allow(dead_code)]

mod apple;
mod stub;

use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::apple::ApplePowerRecorder;

pub trait PowerRecorder: Send {
    fn stop(&self) -> Option<ChatReplyPowerStats>;
}

impl dyn PowerRecorder {
    pub fn start() -> Box<dyn PowerRecorder> {
        #[cfg(target_vendor = "apple")]
        {
            Box::new(ApplePowerRecorder::new()) as Box<dyn PowerRecorder>
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            Box::new(StubPowerRecorder::new()) as Box<dyn PowerRecorder>
        }
    }
}
