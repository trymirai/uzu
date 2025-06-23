use crate::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDataType {
    Float32,
    Float16,
    BFloat16,
}

impl KernelDataType {
    pub fn function_name_suffix(&self) -> &'static str {
        match self {
            KernelDataType::Float32 => "float",
            KernelDataType::Float16 => "half",
            KernelDataType::BFloat16 => "bfloat",
        }
    }
}

impl Into<DataType> for KernelDataType {
    fn into(self) -> DataType {
        match self {
            KernelDataType::BFloat16 => DataType::BF16,
            KernelDataType::Float16 => DataType::F16,
            KernelDataType::Float32 => DataType::F32,
        }
    }
}

impl From<DataType> for KernelDataType {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::BF16 => KernelDataType::BFloat16,
            DataType::F16 => KernelDataType::Float16,
            DataType::F32 => KernelDataType::Float32,
            _ => panic!("Unsupported data type: {0:?}", dtype),
        }
    }
}
