use crate::backends::common::{Backend, CpuBuffer, GpuBuffer};

pub trait GlobalGpuBuffer: GpuBuffer<Backend: Backend<GlobalGpuBuffer = Self>> {}

pub trait GlobalBufferCpuGpu: CpuBuffer<Backend: Backend<GlobalBufferCpuGpu = Self>> {
    fn handoff(self) -> <Self::Backend as Backend>::GlobalGpuBuffer;
}
