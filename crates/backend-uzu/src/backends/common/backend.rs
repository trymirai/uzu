use std::{error::Error, fmt::Debug, time::Duration};

use super::{CommandBuffer, Context, DenseBuffer, Kernels, SparseBuffer};

pub trait SharedEvent: Clone + Send + Sync + Unpin {
    fn signaled_value(&self) -> u64;

    fn wait_until_signaled_value_timeout_ms(
        &self,
        value: u64,
        timeout_ms: u64,
    ) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
        let mut spins = 0_u64;
        while self.signaled_value() < value {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::hint::spin_loop();
            spins += 1;
            if spins.is_multiple_of(1024) {
                std::thread::yield_now();
            }
        }
        true
    }

    fn signal(
        &self,
        value: u64,
    );
}

pub trait Backend: Debug + Clone + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DenseBuffer: DenseBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type Kernels: Kernels<Backend = Self>;
    type SharedEvent: SharedEvent;
    type Error: Error + Debug;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_ALLOCATION_ALIGNMENT: usize;
    const ALLOCATION_GRANULARITY: usize;
    const MAX_INLINE_BYTES: usize;
}
