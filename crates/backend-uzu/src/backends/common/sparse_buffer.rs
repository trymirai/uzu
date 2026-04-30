use backend_uzu::backends::common::Backend;

use crate::backends::common::Buffer;

pub trait SparseBuffer: Buffer {
    type Backend: Backend<SparseBuffer = Self>;
}
