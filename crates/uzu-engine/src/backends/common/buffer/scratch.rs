use crate::backends::common::{Backend, Buffer};

pub trait ScratchBuffer: Buffer<Backend: Backend<ScratchBuffer = Self>> {}
