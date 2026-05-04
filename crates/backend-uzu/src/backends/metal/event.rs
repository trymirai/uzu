use metal::prelude::*;

use super::Metal;
use crate::backends::common::Event;

impl Event for Retained<ProtocolObject<dyn MTLEvent>> {
    type Backend = Metal;
}
