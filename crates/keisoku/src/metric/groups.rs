use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct IoReportGroups: u8 {
        const ENERGY_MODEL = 1 << 0;
        const CPU_STATS = 1 << 1;
        const GPU_STATS = 1 << 2;
        const AMC_STATS = 1 << 3;
        const PMP = 1 << 4;
    }
}
