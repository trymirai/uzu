use super::{DramFlow, FlowKind};

pub struct DramRead;

impl FlowKind for DramRead {
    const FLOW: DramFlow = DramFlow::DramRead;
    const TYPE_BIT_BYTES: u128 = 1 << 5;
    const TYPE_BIT_HIST: u128 = 1 << 7;
}
