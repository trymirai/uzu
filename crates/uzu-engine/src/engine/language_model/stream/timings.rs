use std::sync::mpsc::Sender;

use crate::backends::common::CommandBufferTiming;

pub struct Timings {
    pub sender: Sender<CommandBufferTiming>,
    pub spans: bool,
}
