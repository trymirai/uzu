use syn::Ident;

use super::{FunctionParameterType, canonicalize_type_text};
use crate::common::{KernelParameterType, enum_paths::EnumPaths, kernel::KernelParameter};

#[derive(PartialEq, Debug)]
pub struct FunctionParameter {
    pub name: Ident,
    pub ty: FunctionParameterType,
}

impl FunctionParameter {
    pub fn to_kernel_parameter(
        &self,
        data_types: &[Box<str>],
        enum_paths: &EnumPaths,
    ) -> KernelParameter {
        KernelParameter {
            name: self.name.to_string().into_boxed_str(),
            ty: match &self.ty {
                FunctionParameterType::Type => KernelParameterType::types(data_types.iter().cloned()),
                FunctionParameterType::Value(ty) => {
                    KernelParameterType::Value(canonicalize_type_text(ty, enum_paths).into_boxed_str())
                },
            },
        }
    }
}
