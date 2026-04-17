use metal::MTLEvent;
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::Event;

impl Event for Retained<ProtocolObject<dyn MTLEvent>> {
    type Backend = Metal;
}
