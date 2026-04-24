use std::collections::HashMap;

use anyhow::{Context, bail};
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
}

impl SlangKernel {
    pub fn from_reflection(toplevel_decl: &Decl) -> anyhow::Result<Option<Self>> {
        let (generic_decl, function_decl) = if toplevel_decl.kind() == DeclKind::Generic {
            (Some(toplevel_decl), toplevel_decl.as_generic().unwrap().inner_decl().unwrap())
        } else {
            (None, toplevel_decl)
        };

        if function_decl.kind() != DeclKind::Func {
            return Ok(None);
        }

        let function = function_decl.as_function().unwrap();
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

        let name = function.name().unwrap().to_string();

        let arguments = function
            .parameters()
            .map(SlangArgument::from_reflection)
            .collect::<anyhow::Result<Vec<SlangArgument>>>()
            .context("cannot collect arguments")?;

        Ok(Some(SlangKernel {
            parameters,
            constraints,
            public,
            name,
            arguments,
        }))
    }

    pub fn variants(&self) -> impl Iterator<Item = Vec<(&str, &str)>> {
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
            .filter(|variant| constraints::satisfied(&variant, &self.constraints))
    }

    pub fn to_common(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Option<Kernel>> {
        if !self.public {
            return Ok(None);
        }

        Ok(Some(Kernel {
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
                    .filter_map(|argument| {
                        SlangArgument::to_common_argument(argument, gpu_type_map, &mut indirect_used)
                            .with_context(|| format!("cannot convert argument {} to common argument", argument.name))
                            .transpose()
                    })
                    .collect::<anyhow::Result<Box<[KernelArgument]>>>()?
            },
        }))
    }
}
