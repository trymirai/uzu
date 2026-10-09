use quote::ToTokens;
use syn::{Expr, Ident};

use super::{FunctionArgumentType, canonicalize_type_text};
use crate::common::{
    KernelParameterType,
    enum_paths::EnumPaths,
    identifiers::ArgumentName,
    kernel::{KernelArgument, KernelArgumentType, KernelParameter},
};

#[derive(PartialEq, Debug)]
pub struct FunctionArgument {
    pub name: Ident,
    pub conditional: Option<Expr>,
    pub ty: FunctionArgumentType,
}

impl FunctionArgument {
    pub fn to_kernel_argument(
        &self,
        enum_paths: &EnumPaths,
    ) -> Option<KernelArgument> {
        Some(KernelArgument {
            name: ArgumentName::from(self.name.to_string()),
            conditional: self.conditional.is_some(),
            ty: match &self.ty {
                FunctionArgumentType::Buffer(access) => KernelArgumentType::Buffer(access.clone()),
                FunctionArgumentType::Constant(ty, None) => KernelArgumentType::Constant(
                    format!("&[{}]", canonicalize_type_text(ty, enum_paths)).into_boxed_str(),
                ),
                FunctionArgumentType::Constant(ty, Some(size)) => KernelArgumentType::Constant(
                    format!("&[{}; {}]", canonicalize_type_text(ty, enum_paths), size.to_token_stream(),)
                        .into_boxed_str(),
                ),
                FunctionArgumentType::Scalar(ty) => {
                    KernelArgumentType::Constant(canonicalize_type_text(ty, enum_paths).into_boxed_str())
                },
                FunctionArgumentType::Specialization(_) => {
                    return None;
                },
            },
        })
    }

    pub fn to_kernel_parameter(
        &self,
        enum_paths: &EnumPaths,
    ) -> Option<KernelParameter> {
        Some(KernelParameter {
            name: self.name.to_string().into_boxed_str(),
            ty: match &self.ty {
                FunctionArgumentType::Specialization(ty) => {
                    KernelParameterType::Value(canonicalize_type_text(ty, enum_paths).into_boxed_str())
                },
                _ => {
                    return None;
                },
            },
        })
    }
}
