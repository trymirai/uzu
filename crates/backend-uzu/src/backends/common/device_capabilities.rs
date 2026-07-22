use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        const SPARSE_BUFFERS = 1 << 0;
        const HARDWARE_INT8_MATMUL = 1 << 1;
    }
}
