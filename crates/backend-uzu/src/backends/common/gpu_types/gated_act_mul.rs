#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GatedActMulOp {
    FullPrecision,
    Quantize,
    QuantizeWithGroupSums,
}
