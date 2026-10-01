use crate::backends::common::{Backend, BufferCpuAccessible};

pub trait GlobalBuffer: BufferCpuAccessible<Backend: Backend<GlobalBuffer = Self>> {}
