use crate::backends::common::{Backend, CpuBuffer, GpuBuffer};

pub trait GlobalGpuBuffer: GpuBuffer<Backend: Backend<GlobalGpuBuffer = Self>> {}

pub trait GlobalBufferCpuGpu: CpuBuffer<Backend: Backend<GlobalBufferCpuGpu = Self>> {
    fn handoff(self) -> Result<<Self::Backend as Backend>::GlobalGpuBuffer, <Self::Backend as Backend>::Error>;
}
