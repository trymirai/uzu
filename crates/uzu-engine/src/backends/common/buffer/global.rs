use crate::backends::common::{Backend, BufferCpuAccessible};

pub trait GlobalBuffer: BufferCpuAccessible<Backend: Backend<GlobalBuffer = Self>> {}

pub trait GlobalBufferInitializer: BufferCpuAccessible<Backend: Backend<GlobalBufferInitializer = Self>> {
    fn end_initialization(self) -> <Self::Backend as Backend>::GlobalBuffer;
}
