use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        const GLOBAL_BUFFERS_CPU_ACCESSIBLE = 1 << 0;
        const SPARSE_BUFFERS = 1 << 1;
        /// Speculation runs on devices that the speculator's tree shapes do not list, with the listed shape of the
        /// smallest tree budget.
        const UNTUNED_SPECULATION = 1 << 2;
        /// Tensor cores verify a 16-node speculation tree about as fast as smaller ones, so devices the speculator's
        /// tree shapes do not list speculate with its tensor-core shape.
        const TENSOR_CORE_SPECULATION = 1 << 3;
    }
}
