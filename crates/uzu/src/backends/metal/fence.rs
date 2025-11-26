use std::cell::RefCell;

use metal::{Device as MTLDevice, Fence as MTLFence};

/// Registry for fence management between compute encoders.
/// Required for untracked heap resources where Metal doesn't auto-sync.
pub struct FenceRegistry {
    device: MTLDevice,
    previous_fence: RefCell<Option<MTLFence>>,
}

impl FenceRegistry {
    pub fn new(device: MTLDevice) -> Self {
        Self {
            device,
            previous_fence: RefCell::new(None),
        }
    }

    /// Create a new fence
    pub fn new_fence(&self) -> MTLFence {
        self.device.new_fence()
    }

    /// Take the previous fence (for waiting)
    pub fn take_previous(&self) -> Option<MTLFence> {
        self.previous_fence.borrow_mut().take()
    }

    /// Set current fence (for next encoder to wait on)
    pub fn set_current(
        &self,
        fence: MTLFence,
    ) {
        *self.previous_fence.borrow_mut() = Some(fence);
    }
}
