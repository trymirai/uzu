use std::ptr::NonNull;

use metal::{MTLDataType, MTLFunctionConstantValues};

use crate::backends::common::gpu_types::{
    QuantizationMode, QuantizedFormat,
    unified_gemm::{
        GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmWeightPrologueKind, QuantizationMethod
    },
};

pub trait FunctionConstantValueType {
    const DATA_TYPE: MTLDataType;
}

impl FunctionConstantValueType for bool {
    const DATA_TYPE: MTLDataType = MTLDataType::Bool;
}

impl FunctionConstantValueType for f32 {
    const DATA_TYPE: MTLDataType = MTLDataType::Float;
}

impl FunctionConstantValueType for i32 {
    const DATA_TYPE: MTLDataType = MTLDataType::Int;
}

impl FunctionConstantValueType for u32 {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for GemmInputPrologueKind {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for GemmWeightPrologueKind {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for GemmComputeKind {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for GemmOutputTransformKind {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for GemmAlignment {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for QuantizationMode {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

impl FunctionConstantValueType for QuantizationMethod {
    const DATA_TYPE: MTLDataType = MTLDataType::UInt;
}

/// Extension trait providing ergonomic `set_value` methods for `MTLFunctionConstantValues`.
pub trait FunctionConstantValuesSetValue {
    /// Sets a single supported Rust value as a function constant.
    fn set_value<T: FunctionConstantValueType>(
        &self,
        value: &T,
        index: usize,
    );
}

impl FunctionConstantValuesSetValue for MTLFunctionConstantValues {
    fn set_value<T: FunctionConstantValueType>(
        &self,
        value: &T,
        index: usize,
    ) {
        let ptr = NonNull::from(value).cast();
        self.set_constant_value_type_at_index(ptr, T::DATA_TYPE, index);
    }
}
