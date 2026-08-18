#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedActMulOp {
    FullPrecision,
    Quantize,
    QuantizeWithGroupSums,
}
