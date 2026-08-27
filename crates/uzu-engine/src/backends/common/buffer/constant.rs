use crate::backends::common::{Backend, BufferCpuAccessible};

pub trait ConstantBuffer: BufferCpuAccessible<Backend: Backend<ConstantBuffer = Self>> {}
