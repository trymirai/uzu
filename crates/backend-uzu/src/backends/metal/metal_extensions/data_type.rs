use crate::data_type::DataType;

pub trait MetalDataTypeExt {
    fn metal_type(&self) -> &'static str;

    /// The Metal spelling, or `None` for a type no shader has one for. Generated
    /// `entry_name()`s use this: a key is also a question ("was this variant built?"),
    /// and a question about an unsupported type has an answer rather than a panic.
    fn try_metal_type(&self) -> Option<&'static str>;
}

impl MetalDataTypeExt for DataType {
    fn metal_type(&self) -> &'static str {
        self.try_metal_type().unwrap_or_else(|| panic!("Unsupported data type: {0:?}", self))
    }

    fn try_metal_type(&self) -> Option<&'static str> {
        match self {
            DataType::F16 => Some("half"),
            DataType::BF16 => Some("bfloat"),
            DataType::F32 => Some("float"),
            _ => None,
        }
    }
}
