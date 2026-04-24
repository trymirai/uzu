use std::collections::HashMap;

use anyhow::{Context, anyhow, bail};
use shader_slang::reflection::{Decl, UserAttribute};

use crate::{
    common::kernel::{KernelParameter, KernelParameterType},
    slang::types::slang2rust,
};

#[derive(Debug)]
pub enum SlangParameterType {
    Type {
        variants: Vec<String>,
    },
    GroupShared {
        value_type: String,
        length: String,
    },
    Value {
        value_type: String,
        variants: Vec<String>,
    },
}

#[derive(Debug)]
pub struct SlangParameter {
    pub name: String,
    pub ty: SlangParameterType,
}

impl SlangParameter {
    pub fn from_reflection(
        function_attributes: &[&UserAttribute],
        generic_decl: Option<&Decl>,
    ) -> anyhow::Result<Vec<Self>> {
        enum ParameterAttributeType<'a> {
            Variants {
                variants: Vec<&'a str>,
            },
            GroupShared {
                length: &'a str,
            },
        }

        struct ParameterAttribute<'a> {
            name: &'a str,
            ty: ParameterAttributeType<'a>,
        }

        let parameters_attributes = function_attributes
            .iter()
            .map(|user_attribute| {
                Ok(match user_attribute.name() {
                    Some("Variants") => {
                        if user_attribute.argument_count() != 2 {
                            bail!("expected 2 arguments, found {}", user_attribute.argument_count());
                        }

                        let name = user_attribute
                            .argument_value_string(0)
                            .context("cannot get name argument string value")?
                            .trim();

                        let variants = user_attribute
                            .argument_value_string(1)
                            .context("cannot get variants argument string value")?
                            .split(',')
                            .map(|variant| variant.trim())
                            .collect::<Vec<&str>>();

                        Some(ParameterAttribute {
                            name,
                            ty: ParameterAttributeType::Variants {
                                variants,
                            },
                        })
                    },
                    Some("GroupShared") => {
                        if user_attribute.argument_count() != 2 {
                            bail!("expected 2 arguments, found {}", user_attribute.argument_count());
                        }

                        let name = user_attribute
                            .argument_value_string(0)
                            .context("cannot get name argument string value")?
                            .trim();

                        let length = user_attribute
                            .argument_value_string(1)
                            .context("cannot get length argument string value")?
                            .trim();

                        Some(ParameterAttribute {
                            name,
                            ty: ParameterAttributeType::GroupShared {
                                length,
                            },
                        })
                    },
                    _ => None,
                })
            })
            .filter_map(|attr| attr.transpose())
            .collect::<anyhow::Result<Vec<_>>>()
            .context("failed to collect parameter attributes")?;

        enum ParameterGenericType {
            Type {
                constraints: Vec<String>,
            },
            Value {
                value_type: String,
            },
        }

        struct ParameterGeneric<'a> {
            name: &'a str,
            ty: ParameterGenericType,
        }

        let parameters_generics = generic_decl
            .map(|generic_decl| -> anyhow::Result<_> {
                let generic = generic_decl.as_generic().unwrap();

                let mut generics_unordered = generic
                    .type_parameters()
                    .map(|type_parameter| {
                        (
                            type_parameter.name().unwrap(),
                            ParameterGenericType::Type {
                                constraints: (0..generic.type_parameter_constraint_count(type_parameter))
                                    .map(|i| {
                                        generic
                                            .type_parameter_constraint_by_index(type_parameter, i)
                                            .unwrap()
                                            .full_name()
                                            .unwrap()
                                            .as_str()
                                            .unwrap()
                                            .to_string()
                                    })
                                    .collect(),
                            },
                        )
                    })
                    .chain(generic.value_parameters().map(|value_parameter| {
                        (
                            value_parameter.name().unwrap(),
                            ParameterGenericType::Value {
                                value_type: value_parameter
                                    .ty()
                                    .unwrap()
                                    .full_name()
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string(),
                            },
                        )
                    }))
                    .collect::<HashMap<&str, ParameterGenericType>>();

                let mut generics = Vec::with_capacity(generics_unordered.len());

                for generic_name in
                    generic_decl.children().filter_map(|generic_parameter_decl| generic_parameter_decl.name())
                {
                    if let Some(generic_type) = generics_unordered.remove(generic_name) {
                        generics.push(ParameterGeneric {
                            name: generic_name,
                            ty: generic_type,
                        });
                    }
                }

                if !generics_unordered.is_empty() {
                    bail!(
                        "generics present in parameters api but not generic decl children: {:?}",
                        generics_unordered.keys().copied().collect::<Vec<&str>>()
                    );
                }

                Ok(generics)
            })
            .transpose()
            .context("failed to collect generic parameters")?
            .unwrap_or_default();

        let parameters_attributes_names =
            parameters_attributes.iter().map(|parameter_attribute| parameter_attribute.name).collect::<Vec<&str>>();
        let parameters_generics_names =
            parameters_generics.iter().map(|parameter_generic| parameter_generic.name).collect::<Vec<&str>>();
        if parameters_attributes_names != parameters_generics_names {
            bail!(
                "variants attributes {:?} do not match function generics {:?}",
                parameters_attributes_names,
                parameters_generics_names,
            );
        }

        let parameters = parameters_attributes
            .into_iter()
            .zip(parameters_generics)
            .map(|(parameter_attribute, parameter_generic)| {
                Ok(SlangParameter {
                    name: parameter_generic.name.to_string(),
                    ty: match (parameter_generic.ty, parameter_attribute.ty) {
                        (
                            ParameterGenericType::Type {
                                constraints: _,
                            },
                            ParameterAttributeType::Variants {
                                variants,
                            },
                        ) => SlangParameterType::Type {
                            variants: variants.into_iter().map(|variant| variant.to_string()).collect(),
                        },
                        (
                            ParameterGenericType::Type {
                                constraints,
                            },
                            ParameterAttributeType::GroupShared {
                                length,
                            },
                        ) => {
                            if constraints.len() != 1 {
                                bail!(
                                    "type generic associated with GroupShared must have exactly 1 constraint, found {}",
                                    constraints.len()
                                );
                            }
                            let constraint = constraints[0].as_str();

                            SlangParameterType::GroupShared {
                                value_type: constraint
                                    .strip_prefix("IGroupShared<")
                                    .and_then(|s| s.strip_suffix(">"))
                                    .ok_or_else(|| anyhow!("{constraint} doesn't match IGroupShared<T> pattern"))?
                                    .to_string(),
                                length: length.to_string(),
                            }
                        },
                        (
                            ParameterGenericType::Value {
                                value_type,
                            },
                            ParameterAttributeType::Variants {
                                variants,
                            },
                        ) => SlangParameterType::Value {
                            value_type,
                            variants: variants.into_iter().map(|variant| variant.to_string()).collect(),
                        },
                        (
                            ParameterGenericType::Value {
                                value_type: _,
                            },
                            ParameterAttributeType::GroupShared {
                                length: _,
                            },
                        ) => bail!("value generic cannot be the target of GroupShared annotation"),
                    },
                })
            })
            .collect::<anyhow::Result<Vec<SlangParameter>>>()?;

        Ok(parameters)
    }

    pub fn to_common(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Option<KernelParameter>> {
        let name = self.name.clone().into_boxed_str();

        match &self.ty {
            SlangParameterType::Type {
                variants: _,
            } => Ok(Some(KernelParameter {
                name,
                ty: KernelParameterType::Type,
            })),
            SlangParameterType::GroupShared {
                value_type: _,
                length: _,
            } => Ok(None),
            SlangParameterType::Value {
                value_type,
                variants: _,
            } => Ok(Some(KernelParameter {
                name,
                ty: KernelParameterType::Value(
                    slang2rust(value_type, gpu_type_map)
                        .with_context(|| format!("cannot convert {value_type} to rust"))?
                        .to_string()
                        .into_boxed_str(),
                ),
            })),
        }
    }
}
