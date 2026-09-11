mod flow;
mod rail;

pub use flow::{DramFlow, DramRead, DramWrite, FlowKind};
pub use rail::{Ane, Cpu, Gpu, Rail, RailKind, Ram};
