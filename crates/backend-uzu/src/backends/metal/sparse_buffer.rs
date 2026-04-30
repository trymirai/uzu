use backend_uzu::backends::metal::Metal;
use metal::MTLBuffer;
use objc2::__framework_prelude::{ProtocolObject, Retained};

use crate::backends::common::SparseBuffer;

impl SparseBuffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    type Backend = Metal;
}
