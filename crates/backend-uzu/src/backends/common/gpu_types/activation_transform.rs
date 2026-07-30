#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationTransformOp {
    InputRht,
    OutputRht,
    Quantize,
    QuantizeWithGroupSums,
}
