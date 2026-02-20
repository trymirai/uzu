use crate::backends::{
    common::{Backend, CommandBuffer},
    cpu::backend::Cpu,
};

pub struct CpuCommandBuffer;

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    fn with_compute_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::ComputeEncoder) -> T,
    ) -> T {
        todo!()
    }

    fn with_copy_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::CopyEncoder) -> T,
    ) -> T {
        todo!()
    }

    fn submit(&self) {
        todo!()
    }

    fn wait_until_completed(&self) {
        todo!()
    }

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        todo!()
    }
}
