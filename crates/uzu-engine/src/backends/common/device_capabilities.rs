use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        const GLOBAL_BUFFERS_CPU_ACCESSIBLE = 1 << 0;
        const SPARSE_BUFFERS = 1 << 1;
    }
}
