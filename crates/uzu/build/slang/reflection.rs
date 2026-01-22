#![allow(dead_code)] // TODO: finish bindgen

use std::collections::HashMap;

use anyhow::{Context, bail};
use shader_slang::{
    DeclKind, ScalarType, TypeKind,
    reflection::{Decl, Generic, Type, UserAttribute, Variable},
};

use crate::slang::slang_api;

fn variants_for_constraint(
    constraint: &str
) -> Option<&'static [&'static str]> {
    match constraint {
        "__BuiltinFloatingPointType" => Some(&["float", "half", "double"]),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub enum SlangArgumentType {
    Ptr,
    Constant(&'static str),
    Axis(Box<str>, Box<str>),
    Groups(Box<str>),
    Threads(Box<str>),
}

pub struct SlangArgument<'a> {
    variable: &'a Variable,
}

impl<'a> SlangArgument<'a> {
    pub fn new(variable: &'a Variable) -> Self {
        Self {
            variable,
        }
    }

    pub fn name(&self) -> &str {
        self.variable.name()
    }

    pub fn ty(&self) -> &Type {
        self.variable.ty()
    }

    pub fn slang_type(&self) -> String {
        let ty = self.variable.ty();
        ty.full_name()
            .ok()
            .and_then(|b| b.as_str().ok().map(|s| s.to_string()))
            .unwrap_or_else(|| ty.name().to_string())
    }

    pub fn specialized_slang_type(
        &self,
        specialized_generic: &Generic,
    ) -> String {
        let ty = self.variable.ty();
        if let Some(specialized_ty) =
            ty.apply_specializations(specialized_generic)
        {
            specialized_ty
                .full_name()
                .ok()
                .and_then(|b| b.as_str().ok().map(|s| s.to_string()))
                .unwrap_or_else(|| specialized_ty.name().to_string())
        } else {
            self.slang_type()
        }
    }

    pub fn user_attributes(&self) -> impl Iterator<Item = &UserAttribute> {
        self.variable.user_attributes()
    }

    pub fn argument_type(&self) -> anyhow::Result<SlangArgumentType> {
        let ty = self.variable.ty();

        let attrs: HashMap<&str, &UserAttribute> =
            self.variable.user_attributes().map(|ua| (ua.name(), ua)).collect();

        if let Some(axis) = attrs.get("Axis") {
            let total = axis
                .argument_value_string(0)
                .context("Axis missing arg 0")?
                .into();
            let per_group = axis
                .argument_value_string(1)
                .context("Axis missing arg 1")?
                .into();
            Ok(SlangArgumentType::Axis(total, per_group))
        } else if let Some(groups) = attrs.get("Groups") {
            let expr = groups
                .argument_value_string(0)
                .context("Groups missing arg")?
                .into();
            Ok(SlangArgumentType::Groups(expr))
        } else if let Some(threads) = attrs.get("Threads") {
            let expr = threads
                .argument_value_string(0)
                .context("Threads missing arg")?
                .into();
            Ok(SlangArgumentType::Threads(expr))
        } else {
            match ty.kind() {
                TypeKind::Pointer => Ok(SlangArgumentType::Ptr),
                TypeKind::Scalar => {
                    let rust_type = match ty.scalar_type() {
                        ScalarType::Uint32 => "u32",
                        ScalarType::Int32 => "i32",
                        ScalarType::Float32 => "f32",
                        ScalarType::Float64 => "f64",
                        ScalarType::Uint64 => "u64",
                        ScalarType::Int64 => "i64",
                        ScalarType::Float16 => "f16",
                        other => bail!("unsupported scalar type: {:?}", other),
                    };
                    Ok(SlangArgumentType::Constant(rust_type))
                },
                other => bail!(
                    "unsupported parameter type for '{}': kind={:?} name={}",
                    self.name(),
                    other,
                    self.slang_type()
                ),
            }
        }
    }
}

pub struct SlangTypeParameter {
    pub name: String,
    pub variants: &'static [&'static str],
}

pub struct SlangKernelInfo<'a> {
    decl: &'a Decl,
    generic_decl: Option<&'a Decl>,
}

impl<'a> SlangKernelInfo<'a> {
    pub fn from_reflection(decl: &'a Decl) -> anyhow::Result<Option<Self>> {
        let (generic_decl, function_decl) =
            if let DeclKind::Generic = decl.kind() {
                (Some(decl), decl.as_generic().inner_decl())
            } else {
                (None, decl)
            };

        if !matches!(function_decl.kind(), DeclKind::Func) {
            return Ok(None);
        }

        let function = function_decl.as_function();

        if !function.user_attributes().any(|ua| ua.name() == "Kernel") {
            return Ok(None);
        }

        if let Some(gd) = generic_decl {
            for param in slang_api::get_generic_type_parameters(gd) {
                let has_known_constraint = param
                    .constraints
                    .iter()
                    .any(|c| variants_for_constraint(c).is_some());

                if !has_known_constraint {
                    bail!(
                        "generic kernel '{}': type parameter '{}' has no known constraint mapping (constraints: {:?})",
                        function.name(),
                        param.name,
                        param.constraints
                    );
                }
            }
        }

        Ok(Some(SlangKernelInfo {
            decl,
            generic_decl,
        }))
    }

    pub fn name(&self) -> &str {
        self.function_decl().as_function().name()
    }

    pub fn generic_decl(&self) -> Option<&'a Decl> {
        self.generic_decl
    }

    fn function_decl(&self) -> &'a Decl {
        if let Some(gd) = self.generic_decl {
            gd.as_generic().inner_decl()
        } else {
            self.decl
        }
    }

    pub fn arguments(&self) -> impl Iterator<Item = SlangArgument<'a>> {
        self.function_decl().as_function().parameters().map(SlangArgument::new)
    }

    pub fn type_parameters(
        &self
    ) -> impl Iterator<Item = SlangTypeParameter> + '_ {
        self.generic_decl
            .into_iter()
            .flat_map(|gd| slang_api::get_generic_type_parameters(gd))
            .filter_map(|param| {
                let variants = param
                    .constraints
                    .iter()
                    .find_map(|c| variants_for_constraint(c))?;
                Some(SlangTypeParameter {
                    name: param.name,
                    variants,
                })
            })
    }

    pub fn is_generic(&self) -> bool {
        self.generic_decl.is_some()
    }
}
