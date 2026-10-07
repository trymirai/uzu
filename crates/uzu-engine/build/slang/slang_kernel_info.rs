use anyhow::{Context, bail};
use shader_slang::{
    DeclKind,
    reflection::{Decl, Function},
};

use super::{Error, SlangArgument, SlangTypeParameter, slang_api};

pub struct SlangKernelInfo<'a> {
    function: &'a Function,
    name: &'a str,
    generic_decl: Option<&'a Decl>,
    type_parameters: Vec<SlangTypeParameter>,
}

impl<'a> SlangKernelInfo<'a> {
    pub fn from_reflection(decl: &'a Decl) -> Result<Option<Self>, Error> {
        let (generic_decl, function_decl) = if let DeclKind::Generic = decl.kind() {
            (
                Some(decl),
                decl.as_generic()
                    .context("generic declaration has no reflection")?
                    .inner_decl()
                    .context("generic declaration has no inner declaration")?,
            )
        } else {
            (None, decl)
        };

        if !matches!(function_decl.kind(), DeclKind::Func) {
            return Ok(None);
        }
        let function = function_decl.as_function().context("function declaration has no reflection")?;
        if !function.user_attributes().any(|attribute| attribute.name() == Some("Kernel")) {
            return Ok(None);
        }
        let name = function.name().context("Slang kernel has no name")?;
        let mut type_parameters = Vec::new();
        if let Some(generic) = generic_decl {
            for parameter in slang_api::get_generic_type_parameters(generic)? {
                let Some(variants) =
                    parameter.constraints.iter().find_map(|constraint| variants_for_constraint(constraint))
                else {
                    bail!(
                        "generic kernel '{name}': type parameter '{}' has no known constraint mapping (constraints: {:?})",
                        parameter.name,
                        parameter.constraints
                    );
                };
                type_parameters.push(SlangTypeParameter {
                    variants,
                });
            }
        }
        Ok(Some(Self {
            function,
            name,
            generic_decl,
            type_parameters,
        }))
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn generic_decl(&self) -> Option<&'a Decl> {
        self.generic_decl
    }

    pub fn arguments(&self) -> impl Iterator<Item = SlangArgument<'a>> {
        self.function.parameters().map(SlangArgument::new)
    }

    pub fn type_parameters(&self) -> &[SlangTypeParameter] {
        &self.type_parameters
    }
}

fn variants_for_constraint(constraint: &str) -> Option<&'static [&'static str]> {
    match constraint {
        "__BuiltinFloatingPointType" => Some(&["float", "half", "double"]),
        _ => None,
    }
}
