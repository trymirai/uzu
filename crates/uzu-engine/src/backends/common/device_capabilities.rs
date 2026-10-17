use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        const CPU_BUFFER_DIRECT_ACCESS = 1 << 0;
        const SPARSE_BUFFERS = 1 << 1;
    }
}
