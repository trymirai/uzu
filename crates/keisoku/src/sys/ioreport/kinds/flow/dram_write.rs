use super::{DramFlow, FlowKind};

pub struct DramWrite;

impl FlowKind for DramWrite {
    const FLOW: DramFlow = DramFlow::DramWrite;
    const TYPE_BIT_BYTES: u128 = 1 << 6;
    const TYPE_BIT_HIST: u128 = 1 << 8;
}
