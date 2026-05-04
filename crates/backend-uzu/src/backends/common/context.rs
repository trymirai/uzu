use std::{env, path::Path, rc::Rc};

use crate::backends::common::{Allocation, AllocationPool, AllocationType, Backend, CommandBuffer};

pub trait Context: Sized {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error>;

    fn recommended_async_batch_size(
        &self,
        model_path: &Path,
    ) -> usize;

    fn is_high_performance(&self) -> bool;

    fn debug_active(&self) -> bool;

    fn create_command_buffer(
        &self
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Initial, <Self::Backend as Backend>::Error>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::DenseBuffer, <Self::Backend as Backend>::Error>;

    fn create_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType<Self::Backend>,
    ) -> Result<Allocation<Self::Backend>, <Self::Backend as Backend>::Error>;

    fn create_allocation_pool(
        &self,
        reusable: bool,
    ) -> AllocationPool<Self::Backend>;

    fn create_event(&self) -> Result<<Self::Backend as Backend>::Event, <Self::Backend as Backend>::Error>;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error>;

    fn peak_memory_usage(&self) -> Option<usize>;

    fn enable_capture();

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error>;

    fn tf32_enabled() -> bool {
        env::var("UZU_TF32").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
    }
}
