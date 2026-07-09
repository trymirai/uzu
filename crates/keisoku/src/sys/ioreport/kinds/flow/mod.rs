pub(crate) trait FlowKind: 'static {
    const FLOW: DramFlow;
    const TYPE_BIT_BYTES: u128;
    const TYPE_BIT_HIST: u128;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DramFlow {
    DramRead,
    DramWrite,
}

pub struct DramRead;
pub struct DramWrite;

impl FlowKind for DramRead {
    const FLOW: DramFlow = DramFlow::DramRead;
    const TYPE_BIT_BYTES: u128 = 1 << 5;
    const TYPE_BIT_HIST: u128 = 1 << 7;
}

impl FlowKind for DramWrite {
    const FLOW: DramFlow = DramFlow::DramWrite;
    const TYPE_BIT_BYTES: u128 = 1 << 6;
    const TYPE_BIT_HIST: u128 = 1 << 8;
}
