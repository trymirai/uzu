use super::{Rail, RailKind};

pub struct Ane;

impl RailKind for Ane {
    const RAIL: Rail = Rail::Ane;
    const TYPE_BIT: u128 = 1 << 2;
}
