use super::{Rail, RailKind};

pub struct Ram;

impl RailKind for Ram {
    const RAIL: Rail = Rail::Ram;
    const TYPE_BIT: u128 = 1 << 3;
}
