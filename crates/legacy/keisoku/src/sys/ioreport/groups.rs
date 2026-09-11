use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct IoReportGroups: u8 {
        const ENERGY_MODEL = 1 << 0;
        const AMC_STATS = 1 << 1;
        const PMP = 1 << 2;
    }
}
