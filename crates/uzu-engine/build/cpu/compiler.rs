use quote::ToTokens;
use syn::{Expr, Type};

use crate::common::{enum_paths::EnumPaths, kernel::KernelBufferAccess};

#[derive(PartialEq, Debug)]
pub enum FunctionArgumentType {
    Buffer(KernelBufferAccess),
    Constant(Type, Option<Expr>),
    Scalar(Type),
    Specialization(Type),
}

pub fn canonicalize_type_text(
    ty: &Type,
    enum_paths: &EnumPaths,
) -> String {
    let mut canonicalized = ty.clone();
    enum_paths.canonicalize_type(&mut canonicalized);
    canonicalized.to_token_stream().to_string().replace(" :: ", "::")
}

#[derive(PartialEq, Debug)]
pub enum FunctionParameterType {
    Type,
    Value(Type),
}
