use std::collections::HashMap;

use anyhow::{Context, bail, ensure};
use itertools::Itertools;
use shader_slang::{
    DeclKind,
    reflection::{Decl, UserAttribute},
};

use crate::{
    common::{
        constraints,
        kernel::{Kernel, KernelArgument, KernelParameter},
    },
    slang::{
        SlangParameterType,
        reflection::{SlangArgument, SlangParameter},
    },
};

pub struct SlangKernel {
    pub parameters: Vec<SlangParameter>,
    pub constraints: Vec<String>,
    pub public: bool,
    pub name: String,
    pub arguments: Vec<SlangArgument>,
    /// `(phase, condition)` of each host precondition: phase `new` over the parameters, `encode` also over uniforms.
    pub preconditions: Vec<(String, String)>,
    /// The subgroup size, a host expression, the kernel's pipelines require in full subgroups.
    pub required_subgroup_size: Option<String>,
}

impl SlangKernel {
    pub fn from_reflection(
        toplevel_decl: &Decl,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Option<Self>> {
        let (generic_decl, function_decl) = if toplevel_decl.kind() == DeclKind::Generic {
            let generic = toplevel_decl.as_generic().context("generic declaration has no reflection")?;
            (Some(toplevel_decl), generic.inner_decl().context("generic declaration has no inner declaration")?)
        } else {
            (None, toplevel_decl)
        };

        if function_decl.kind() != DeclKind::Func {
            return Ok(None);
        }

        let function = function_decl.as_function().context("function declaration has no reflection")?;
        let function_attributes = function.user_attributes().collect::<Vec<&UserAttribute>>();

        if !function_attributes.iter().any(|user_attribute| user_attribute.name() == Some("Kernel")) {
            return Ok(None);
        };

        let parameters =
            SlangParameter::from_reflection(&function_attributes, generic_decl).context("cannot collect parameters")?;

        let constraints = function_attributes
            .iter()
            .map(|user_attribute| {
                if user_attribute.name() == Some("Constraint") {
                    if user_attribute.argument_count() != 1 {
                        bail!("expected 1 argument, found {}", user_attribute.argument_count());
                    }

                    Ok(Some(
                        user_attribute
                            .argument_value_string(0)
                            .context("cannot get name argument string value")?
                            .trim()
                            .to_string(),
                    ))
                } else {
                    Ok(None)
                }
            })
            .filter_map(|r| r.transpose())
            .collect::<anyhow::Result<Vec<String>>>()
            .context("cannot collect constraints")?;

        let public = function_attributes.iter().any(|user_attribute| user_attribute.name() == Some("Public"));

        let name = function.name().context("kernel has no name")?.to_string();

        let preconditions = function_attributes
            .iter()
            .filter(|user_attribute| user_attribute.name() == Some("Precondition"))
            .map(|user_attribute| {
                let phase = user_attribute.argument_value_string(0).context("precondition has no phase")?;
                let condition = user_attribute.argument_value_string(1).context("precondition has no condition")?;
                ensure!(matches!(phase, "new" | "encode"), "precondition phase {phase} is neither new nor encode");
                ensure!(!condition.trim().is_empty(), "empty {phase} precondition");
                Ok((phase.to_string(), condition.to_string()))
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .context("cannot collect preconditions")?;

        let required_subgroup_size = function_attributes
            .iter()
            .find(|user_attribute| user_attribute.name() == Some("RequiredSubgroupSize"))
            .map(|user_attribute| {
                anyhow::Ok(
                    user_attribute.argument_value_string(0).context("required subgroup size has no size")?.to_string(),
                )
            })
            .transpose()?;

        let arguments = function
            .parameters()
            .map(|variable| SlangArgument::from_reflection(variable, gpu_type_map))
            .collect::<anyhow::Result<Vec<SlangArgument>>>()
            .context("cannot collect arguments")?;

        Ok(Some(SlangKernel {
            parameters,
            constraints,
            public,
            name,
            arguments,
            preconditions,
            required_subgroup_size,
        }))
    }

    pub fn variants(&self) -> impl Iterator<Item = Vec<(&str, &str)>> {
        let evaluator = constraints::Constraints::new(
            self.parameters
                .iter()
                .filter_map(|parameter| match &parameter.ty {
                    SlangParameterType::Type {
                        variants,
                    }
                    | SlangParameterType::Value {
                        value_type: _,
                        variants,
                    } => Some(variants.iter().map(|variant| variant.as_str())),
                    SlangParameterType::GroupShared {
                        value_type: _,
                        length: _,
                    } => None,
                })
                .flatten(),
            &self.constraints,
        );
        self.parameters
            .iter()
            .filter_map(|parameter| match &parameter.ty {
                SlangParameterType::Type {
                    variants,
                }
                | SlangParameterType::Value {
                    value_type: _,
                    variants,
                } => Some(variants.iter().map(move |variant| (parameter.name.as_str(), variant.as_str()))),
                SlangParameterType::GroupShared {
                    value_type: _,
                    length: _,
                } => None,
            })
            .multi_cartesian_product()
            .filter(move |variant| evaluator.satisfied(variant.iter().copied()))
    }

    pub fn to_common(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Kernel> {
        Ok(Kernel {
            name: self.name.clone().into(),
            parameters: self
                .parameters
                .iter()
                .filter_map(|parameter| {
                    SlangParameter::to_common(parameter, gpu_type_map)
                        .with_context(|| format!("cannot convert parameter {} to common parameter", parameter.name))
                        .transpose()
                })
                .chain(self.arguments.iter().filter_map(|argument| {
                    SlangArgument::to_common_parameter(argument, gpu_type_map)
                        .with_context(|| format!("cannot convert argument {} to common parameter", argument.name))
                        .transpose()
                }))
                .collect::<anyhow::Result<Box<[KernelParameter]>>>()?,
            arguments: {
                let mut indirect_used = false;

                self.arguments
                    .iter()
                    .filter_map(|argument| argument.to_common_argument(&mut indirect_used))
                    .collect::<Box<[KernelArgument]>>()
            },
        })
    }
}
