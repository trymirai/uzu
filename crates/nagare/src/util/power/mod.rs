#![allow(dead_code)]

mod apple;
mod rail;
mod stub;

use shoji::types::session::chat::ChatReplyPowerStats;

pub trait PowerRecorder: Send {
    fn begin(&self);
    fn finish(&self) -> Option<ChatReplyPowerStats>;
}

impl dyn PowerRecorder {
    pub fn create() -> Box<dyn PowerRecorder> {
        #[cfg(target_os = "macos")]
        {
            use crate::util::power::apple::ApplePowerRecorder;
            Box::new(ApplePowerRecorder::new()) as Box<dyn PowerRecorder>
        }

        #[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
        {
            use crate::util::power::rail::RailPowerRecorder;
            Box::new(RailPowerRecorder::new()) as Box<dyn PowerRecorder>
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            use crate::util::power::stub::StubPowerRecorder;
            Box::new(StubPowerRecorder::new()) as Box<dyn PowerRecorder>
        }
    }
}
