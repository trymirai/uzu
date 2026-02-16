use super::super::Metal;
use crate::encodable_block::{
    FullPrecisionLinear as GenericFullPrecisionLinear, QuantizedLinear as GenericQuantizedLinear,
};

pub type FullPrecisionLinear = GenericFullPrecisionLinear<Metal>;
pub type QuantizedLinear = GenericQuantizedLinear<Metal>;
