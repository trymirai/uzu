use anyhow::{Context, bail};
use shader_slang::{
    DeclKind,
    reflection::{Decl, Function},
};

use super::{Error, SlangArgument, SlangArgumentType, slang_api};
use crate::common::{
    identifiers::{ArgumentName, KernelName},
    kernel::{Kernel, KernelArgument, KernelArgumentType, KernelParameter, KernelParameterType},
};

pub struct SlangKernelInfo<'a> {
    function: &'a Function,
    name: &'a str,
    public: bool,
    generic_decl: Option<&'a Decl>,
    type_parameters: Vec<(String, &'static [&'static str])>,
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
        let public = function.user_attributes().any(|attribute| attribute.name() == Some("Public"));
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
                type_parameters.push((parameter.name, variants));
            }
        }
        Ok(Some(Self {
            function,
            name,
            public,
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

    pub fn type_parameters(&self) -> impl Iterator<Item = &'static [&'static str]> {
        self.type_parameters.iter().map(|(_, variants)| *variants)
    }

    /// The shared kernel contract, compared against the CPU and Metal descriptors; `None` for private kernels.
    pub fn to_kernel(&self) -> Result<Option<Kernel>, Error> {
        if !self.public {
            return Ok(None);
        }
        let mut parameters = self
            .type_parameters
            .iter()
            .map(|(name, _)| KernelParameter {
                name: name.as_str().into(),
                ty: KernelParameterType::Type,
            })
            .collect::<Vec<_>>();
        let mut arguments = Vec::new();
        for argument in self.arguments() {
            let name = argument.name()?;
            let conditional = argument.condition()?.is_some();
            let ty = match argument.argument_type()? {
                SlangArgumentType::Ptr(access) => KernelArgumentType::Buffer(access),
                SlangArgumentType::Constant(ty) => KernelArgumentType::Constant(ty),
                _ if conditional => {
                    bail!("kernel '{}': Optional argument '{name}' is not a pointer or constant", self.name)
                },
                SlangArgumentType::Specialize(ty) => {
                    parameters.push(KernelParameter {
                        name: name.into(),
                        ty: KernelParameterType::Value(ty),
                    });
                    continue;
                },
                SlangArgumentType::Axis(..) | SlangArgumentType::Groups | SlangArgumentType::Threads(_) => continue,
            };
            arguments.push(KernelArgument {
                name: ArgumentName::from(name),
                conditional,
                ty,
            });
        }
        Ok(Some(Kernel {
            name: KernelName::from(self.name),
            parameters: parameters.into(),
            arguments: arguments.into(),
        }))
    }
}

fn variants_for_constraint(constraint: &str) -> Option<&'static [&'static str]> {
    match constraint {
        "__BuiltinFloatingPointType" => Some(&["float", "half"]),
        "IStorageFloat" => Some(&["float", "half", "bf16"]),
        _ => None,
    }
}
