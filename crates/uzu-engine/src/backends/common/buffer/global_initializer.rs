use crate::backends::common::{Backend, BufferCpuAccessible};

/// Host-written contents of a new global buffer, which become GPU-visible when initialization ends.
pub trait GlobalBufferInitializer: BufferCpuAccessible<Backend: Backend<GlobalBufferInitializer = Self>> {
    fn end_initialization(self) -> Result<<Self::Backend as Backend>::GlobalBuffer, <Self::Backend as Backend>::Error>;
}
