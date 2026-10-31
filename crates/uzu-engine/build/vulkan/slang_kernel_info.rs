use anyhow::{Context, bail, ensure};
use shader_slang::{
    DeclKind,
    reflection::{Decl, Function, Type},
};

use super::{Error, SlangArgument, SlangArgumentType};
use crate::common::{
    gpu_type_paths::GpuTypePaths,
    identifiers::{ArgumentName, KernelName},
    kernel::{Kernel, KernelArgument, KernelArgumentType, KernelParameter, KernelParameterType},
};

pub struct SlangKernelInfo<'a> {
    function: &'a Function,
    name: &'a str,
    public: bool,
    generic_decl: Option<&'a Decl>,
    type_parameters: Vec<(String, &'static [&'static str])>,
    gpu_type_paths: &'a GpuTypePaths,
}

impl<'a> SlangKernelInfo<'a> {
    pub fn from_reflection(
        decl: &'a Decl,
        gpu_type_paths: &'a GpuTypePaths,
    ) -> Result<Option<Self>, Error> {
        let (generic, function_decl) = if let DeclKind::Generic = decl.kind() {
            let generic = decl.as_generic().context("generic declaration has no reflection")?;
            (Some(generic), generic.inner_decl().context("generic declaration has no inner declaration")?)
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
        if let Some(generic) = generic {
            for parameter in generic.type_parameters() {
                let parameter_name = parameter.name().context("type parameter has no name")?;
                let constraints = (0..generic.type_parameter_constraint_count(parameter))
                    .map(|index| generic.type_parameter_constraint_by_index(parameter, index).and_then(Type::name))
                    .collect::<Option<Vec<_>>>()
                    .context("type parameter constraint has no name")?;
                let Some(variants) = constraints.iter().find_map(|constraint| variants_for_constraint(constraint))
                else {
                    bail!(
                        "generic kernel '{name}': type parameter '{parameter_name}' has no known constraint mapping \
                         (constraints: {constraints:?})"
                    );
                };
                type_parameters.push((parameter_name.to_owned(), variants));
            }
        }
        Ok(Some(Self {
            function,
            name,
            public,
            generic_decl: generic.map(|_| decl),
            type_parameters,
            gpu_type_paths,
        }))
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn generic_decl(&self) -> Option<&'a Decl> {
        self.generic_decl
    }

    pub fn arguments(&self) -> impl Iterator<Item = SlangArgument<'a>> {
        self.function.parameters().map(|parameter| SlangArgument::new(parameter, self.gpu_type_paths))
    }

    pub fn type_parameters(&self) -> impl Iterator<Item = &'static [&'static str]> {
        self.type_parameters.iter().map(|(_, variants)| *variants)
    }

    pub fn is_public(&self) -> bool {
        self.public
    }

    /// `(phase, condition)` of every `[[Precondition]]`: phase `new` or `encode`, a nonempty host condition.
    pub fn preconditions(&self) -> Result<Vec<(&'a str, &'a str)>, Error> {
        let attributes = self.function.user_attributes().filter(|attribute| attribute.name() == Some("Precondition"));
        attributes
            .map(|attribute| {
                let phase = attribute.argument_value_string(0).context("Precondition missing phase")?;
                let condition = attribute.argument_value_string(1).context("Precondition missing condition")?;
                ensure!(
                    matches!(phase, "new" | "encode"),
                    "kernel '{}': Precondition phase '{phase}' is neither 'new' nor 'encode'",
                    self.name
                );
                ensure!(!condition.trim().is_empty(), "kernel '{}': empty {phase} Precondition", self.name);
                Ok((phase, condition))
            })
            .collect()
    }

    /// The `[[RequiredSubgroupSize]]` host expression: the subgroup size the kernel's pipelines require, if any.
    pub fn required_subgroup_size(&self) -> Result<Option<&'a str>, Error> {
        self.function
            .user_attributes()
            .find(|attribute| attribute.name() == Some("RequiredSubgroupSize"))
            .map(|attribute| attribute.argument_value_string(0).context("RequiredSubgroupSize missing size"))
            .transpose()
    }

    /// The common kernel signature; only public ones are compared against the other backends'.
    pub fn to_kernel(&self) -> Result<Kernel, Error> {
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
                SlangArgumentType::Constant(ty) => KernelArgumentType::Value(ty),
                SlangArgumentType::Specialize(ty) => {
                    parameters.push(KernelParameter {
                        name: name.into(),
                        ty: KernelParameterType::Value(ty),
                    });
                    continue;
                },
                _ if conditional => {
                    bail!(
                        "kernel '{}': Optional argument '{name}' is not a pointer, constant or specialization",
                        self.name
                    )
                },
                SlangArgumentType::Axis(..) | SlangArgumentType::Groups(_) | SlangArgumentType::Threads(_) => continue,
            };
            arguments.push(KernelArgument {
                name: ArgumentName::from(name),
                conditional,
                ty,
            });
        }
        Ok(Kernel {
            name: KernelName::from(self.name),
            parameters: parameters.into(),
            arguments: arguments.into(),
        })
    }
}

fn variants_for_constraint(constraint: &str) -> Option<&'static [&'static str]> {
    match constraint {
        "IStorageFloat" => Some(&["float", "half", "bf16"]),
        "IFloat32" => Some(&["float"]),
        "IModelFloat" => Some(&["float", "bf16"]),
        "IModelFloat16" => Some(&["bf16"]),
        _ => None,
    }
}
