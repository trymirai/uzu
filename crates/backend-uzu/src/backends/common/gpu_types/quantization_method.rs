use std::fmt;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QuantizationMethod {
    ScaleBias,
    ScaleZeroPoint,
    ScaleSymmetric,
}

impl fmt::Display for QuantizationMethod {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
