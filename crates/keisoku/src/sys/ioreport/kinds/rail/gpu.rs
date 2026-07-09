use super::{Rail, RailKind};

pub struct Gpu;

impl RailKind for Gpu {
    const RAIL: Rail = Rail::Gpu;
    const TYPE_BIT: u128 = 1 << 1;
}
