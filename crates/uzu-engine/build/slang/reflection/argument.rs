use std::collections::{HashMap, hash_map::Entry};

use anyhow::{Context, anyhow, bail, ensure};
use shader_slang::{
    TypeKind,
    reflection::{UserAttribute, Variable},
};

use crate::{
    common::{
        identifiers::ArgumentName,
        kernel::{KernelArgument, KernelArgumentType, KernelBufferAccess, KernelParameter, KernelParameterType},
    },
    slang::{reflection::attribute_arguments::string_arguments, types::slang2rust},
};

#[derive(Debug)]
pub enum SlangArgumentType {
    Buffer {
        condition: Option<String>,
    },
    Uniform {
        condition: Option<String>,
    },
    Specialize {
        condition: Option<String>,
    },
    Axis {
        threads: String,
        threads_in_group: String,
    },
    Groups {
        groups: String,
    },
    Threads {
        threads: String,
    },
    ThreadContext,
}

#[derive(Debug)]
pub struct SlangArgument {
    pub name: String,
    pub slang_type: String,
    pub argument_type: SlangArgumentType,
    pub binding_type: Option<KernelArgumentType>,
    pub pipeline_variants: bool,
}

impl SlangArgument {
    pub fn from_reflection(
        variable: &Variable,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        let name = variable.name().context("argument has no name")?.to_string();

        let ty = variable.ty().with_context(|| format!("argument {name} has no type"))?;

        let type_name = ty.name().with_context(|| format!("type of argument {name} has no name"))?;
        let type_full_name = ty.full_name()?.as_str()?.to_string();

        let mut user_attributes = variable.user_attributes().try_fold(HashMap::new(), |mut acc, user_attributes| {
            match acc.entry(user_attributes.name().context("attribute has no name")?) {
                Entry::Occupied(occupied) => Err(anyhow!("duplicate {} attribute", occupied.key())),
                Entry::Vacant(vacant) => {
                    vacant.insert(user_attributes);
                    Ok(acc)
                },
            }
        })?;

        let get_condition = |user_attributes: &mut HashMap<&str, &UserAttribute>| -> anyhow::Result<Option<String>> {
            let Some(optional_attribute) = user_attributes.remove("Optional") else {
                return Ok(None);
            };

            let [condition] = string_arguments::<1>(optional_attribute)?;
            Ok(Some(condition.to_string()))
        };

        let no_attributes = |user_attributes: &HashMap<&str, &UserAttribute>, name: &str| {
            if !user_attributes.is_empty() {
                bail!("{name} has unexpected attributes: {:?}", user_attributes.keys().copied().collect::<Vec<&str>>());
            }

            Ok(())
        };

        let mut binding_type = None;
        let mut pipeline_variants = false;
        let argument_type = if let Some(specialize_attribute) = user_attributes.remove("Specialize") {
            string_arguments::<0>(specialize_attribute)?;

            let condition = get_condition(&mut user_attributes)?;
            no_attributes(&user_attributes, "specialize")?;

            SlangArgumentType::Specialize {
                condition,
            }
        } else if let Some(axis_attribute) = user_attributes.remove("Axis") {
            let [threads, threads_in_group] = string_arguments::<2>(axis_attribute)?.map(str::to_string);

            no_attributes(&user_attributes, "axis")?;

            SlangArgumentType::Axis {
                threads,
                threads_in_group,
            }
        } else if let Some(groups_attribute) = user_attributes.remove("Groups") {
            let [groups] = string_arguments::<1>(groups_attribute)?.map(str::to_string);

            no_attributes(&user_attributes, "groups")?;

            SlangArgumentType::Groups {
                groups,
            }
        } else if let Some(threads_attribute) = user_attributes.remove("Threads") {
            let [threads] = string_arguments::<1>(threads_attribute)?.map(str::to_string);

            no_attributes(&user_attributes, "threads")?;

            SlangArgumentType::Threads {
                threads,
            }
        } else if type_name == "ThreadContext" {
            no_attributes(&user_attributes, "thread context")?;

            SlangArgumentType::ThreadContext
        } else if matches!(ty.kind(), TypeKind::Pointer) {
            let condition = get_condition(&mut user_attributes)?;
            let (pointee, access_type) = type_full_name
                .strip_prefix("Ptr<")
                .and_then(|ty| {
                    ty.strip_suffix(", AddressSpace.Device>")
                        .or_else(|| ty.strip_suffix(", AddressSpace.Device, DefaultDataLayout>"))
                })
                .and_then(|ty| ty.rsplit_once(", Access."))
                .and_then(|(pointee, access)| match access {
                    "Read" => Some((pointee, KernelBufferAccess::Read)),
                    "ReadWrite" => Some((pointee, KernelBufferAccess::ReadWrite)),
                    _ => None,
                })
                .with_context(|| {
                    format!("pointer argument doesn't match Ptr<T, Access, AddressSpace.Device>: {type_full_name}")
                })?;
            binding_type = Some(if user_attributes.remove("Constant").is_some() {
                ensure!(access_type == KernelBufferAccess::Read, "constant pointer {name} is not read-only");
                KernelArgumentType::Value(
                    slang2rust(&format!("{pointee}[]"), gpu_type_map)
                        .with_context(|| format!("cannot convert slang {type_full_name} to rust"))?
                        .into_owned()
                        .into_boxed_str(),
                )
            } else {
                KernelArgumentType::Buffer(access_type)
            });

            no_attributes(&user_attributes, "pointer")?;

            SlangArgumentType::Buffer {
                condition,
            }
        } else if type_name == "StructuredBuffer" {
            let condition = get_condition(&mut user_attributes)?;
            binding_type = Some(if user_attributes.remove("Constant").is_some() {
                let element_type = type_full_name
                    .strip_prefix("StructuredBuffer<")
                    .and_then(|ty| ty.strip_suffix(", DefaultDataLayout>"))
                    .with_context(|| format!("value buffer argument doesn't match StructuredBuffer<T, DefaultDataLayout>: {type_full_name}"))?;
                KernelArgumentType::Value(
                    slang2rust(&format!("{element_type}[]"), gpu_type_map)
                        .with_context(|| format!("cannot convert slang {type_full_name} to rust"))?
                        .into_owned()
                        .into_boxed_str(),
                )
            } else {
                KernelArgumentType::Buffer(KernelBufferAccess::Read)
            });

            no_attributes(&user_attributes, "buffer (read-only)")?;

            SlangArgumentType::Buffer {
                condition,
            }
        } else if type_name == "RWStructuredBuffer" {
            let condition = get_condition(&mut user_attributes)?;
            binding_type = Some(KernelArgumentType::Buffer(KernelBufferAccess::ReadWrite));

            no_attributes(&user_attributes, "buffer (read-write)")?;

            SlangArgumentType::Buffer {
                condition,
            }
        } else {
            let condition = get_condition(&mut user_attributes)?;
            pipeline_variants = user_attributes.remove("PipelineVariants").is_some();
            ensure!(!pipeline_variants || condition.is_none(), "pipeline variants {name} cannot be optional");

            no_attributes(&user_attributes, "uniform")?;
            binding_type = Some(KernelArgumentType::Value(
                slang2rust(&type_full_name, gpu_type_map)
                    .with_context(|| format!("cannot convert slang {type_full_name} to rust"))?
                    .into_owned()
                    .into_boxed_str(),
            ));

            SlangArgumentType::Uniform {
                condition,
            }
        };

        Ok(SlangArgument {
            name,
            slang_type: type_full_name,
            argument_type,
            binding_type,
            pipeline_variants,
        })
    }

    pub fn rust_type(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Box<str>> {
        Ok(slang2rust(&self.slang_type, gpu_type_map)
            .with_context(|| format!("cannot convert slang {} to rust", self.slang_type))?
            .to_string()
            .into_boxed_str())
    }

    pub fn to_common_parameter(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Option<KernelParameter>> {
        let SlangArgumentType::Specialize {
            condition,
        } = &self.argument_type
        else {
            return Ok(None);
        };

        Ok(Some(KernelParameter {
            name: self.name.clone().into_boxed_str(),
            ty: KernelParameterType::Value(if condition.is_some() {
                format!("Option < {} >", self.rust_type(gpu_type_map)?).into_boxed_str()
            } else {
                self.rust_type(gpu_type_map)?
            }),
        }))
    }

    pub fn to_common_argument(
        &self,
        indirect_used: &mut bool,
    ) -> Option<KernelArgument> {
        if let Some(ty) = &self.binding_type {
            let condition = match &self.argument_type {
                SlangArgumentType::Buffer {
                    condition,
                    ..
                }
                | SlangArgumentType::Uniform {
                    condition,
                } => condition,
                _ => unreachable!(),
            };
            Some(KernelArgument {
                name: ArgumentName::from(self.name.clone()),
                conditional: condition.is_some(),
                ty: ty.clone(),
            })
        } else if let SlangArgumentType::Groups {
            groups,
        } = &self.argument_type
            && groups == "INDIRECT"
            && !*indirect_used
        {
            *indirect_used = true;
            Some(KernelArgument {
                name: ArgumentName::from("__dsl_indirect_dispatch_buffer".to_string()),
                conditional: false,
                ty: KernelArgumentType::Buffer(KernelBufferAccess::Read),
            })
        } else {
            None
        }
    }
}
