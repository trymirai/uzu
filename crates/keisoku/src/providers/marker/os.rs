use super::{constant::ConstantMetric, typelist::Metric};
use crate::sources::Sources;

pub struct Os;

impl Metric for Os {
    type Value = String;
    const TYPE_BIT: u128 = 1 << 0;
}

impl ConstantMetric for Os {
    fn read(sources: &mut Sources) -> String {
        sources.os_version()
    }
}
