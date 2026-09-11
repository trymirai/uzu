use metal::MTLGPUFamily;

pub trait GpuFamilyExt: Sized {
    /// All Apple GPU family cases, oldest to newest.
    fn all_cases() -> [Self; 10];
}

impl GpuFamilyExt for MTLGPUFamily {
    fn all_cases() -> [Self; 10] {
        [
            Self::Apple1,
            Self::Apple2,
            Self::Apple3,
            Self::Apple4,
            Self::Apple5,
            Self::Apple6,
            Self::Apple7,
            Self::Apple8,
            Self::Apple9,
            Self::Apple10,
        ]
    }
}
