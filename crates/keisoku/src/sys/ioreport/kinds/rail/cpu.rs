use super::{Rail, RailKind};

pub struct Cpu;

impl RailKind for Cpu {
    const RAIL: Rail = Rail::Cpu;
    const TYPE_BIT: u128 = 1 << 0;
}
