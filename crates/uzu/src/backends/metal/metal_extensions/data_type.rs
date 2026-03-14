use crate::DataType;

pub trait MetalDataTypeExt {
    fn metal_type(&self) -> &'static str;
}

impl MetalDataTypeExt for DataType {
    fn metal_type(&self) -> &'static str {
        match self {
            DataType::I8 => "int8_t",
            DataType::I16 => "short",
            DataType::I32 => "int",
            DataType::F16 => "half",
            DataType::BF16 => "bfloat",
            DataType::F32 => "float",
            _ => panic!("Unsupported data type: {0:?}", self),
        }
    }
}
