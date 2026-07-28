use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ActivationTransformOp: u32 {
        const INPUT_RHT  = 1 << 0;
        const OUTPUT_RHT = 1 << 1;
        const QUANTIZE   = 1 << 2;
        const GROUP_SUMS = 1 << 3;
    }
}

impl ActivationTransformOp {
    pub fn validate(self) -> Self {
        assert!(
            self.contains(Self::INPUT_RHT) ^ self.contains(Self::OUTPUT_RHT),
            "exactly one of INPUT_RHT / OUTPUT_RHT is required, got {self:?}"
        );
        assert!(
            !self.contains(Self::QUANTIZE) || self.contains(Self::INPUT_RHT),
            "QUANTIZE requires INPUT_RHT, got {self:?}"
        );
        assert!(
            !self.contains(Self::GROUP_SUMS) || self.contains(Self::QUANTIZE),
            "GROUP_SUMS requires QUANTIZE, got {self:?}"
        );
        self
    }
}
