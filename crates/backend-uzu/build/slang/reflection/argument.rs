use std::{
    borrow::Cow,
    collections::{HashMap, hash_map::Entry},
};

use anyhow::{Context, anyhow, bail};
use shader_slang::reflection::{UserAttribute, Variable};

use crate::{
    common::kernel::{KernelArgument, KernelArgumentType, KernelBufferAccess, KernelParameter, KernelParameterType},
    slang::types::slang2rust,
};

#[derive(Debug)]
pub enum SlangBufferAccess {
    Read {
        is_constant: bool,
    },
    ReadWrite,
}

#[derive(Debug)]
pub enum SlangArgumentType {
    Buffer {
        access_type: SlangBufferAccess,
        condition: Option<String>,
    },
    Constant {
        condition: Option<String>,
    },
    Specialize,
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
}

impl SlangArgument {
    pub fn from_reflection(variable: &Variable) -> anyhow::Result<Self> {
        let name = variable.name().unwrap().to_string();

        let ty = variable.ty().unwrap();

        let type_name = ty.name().unwrap();
        let type_full_name = ty.full_name().unwrap().as_str().unwrap().to_string();

        let mut user_attributes = variable.user_attributes().try_fold(HashMap::new(), |mut acc, user_attributes| {
            match acc.entry(user_attributes.name().unwrap()) {
                Entry::Occupied(occupied) => Err(anyhow!("duplicate {} attribute", occupied.key())),
                Entry::Vacant(vacant) => {
                    vacant.insert(user_attributes);
                    Ok(acc)
                },
            }
        })?;

        let get_condition = |user_attributes: &mut HashMap<&str, &UserAttribute>| {
            let Some(optional_attribute) = user_attributes.remove("Optional") else {
                return Ok(None);
            };

            if optional_attribute.argument_count() != 1 {
                bail!("optional expected 1 argument, found {}", optional_attribute.argument_count());
            }

            Ok(Some(optional_attribute.argument_value_string(0).unwrap().to_string()))
        };

        let no_attributes = |user_attributes: &HashMap<&str, &UserAttribute>, name: &str| {
            if !user_attributes.is_empty() {
                bail!("{name} has unexpected attributes: {:?}", user_attributes.keys().copied().collect::<Vec<&str>>());
            }

            Ok(())
        };

        let argument_type = if let Some(specialize_attribute) = user_attributes.remove("Specialize") {
            if specialize_attribute.argument_count() != 0 {
                bail!("specialize expected no argument, found {}", specialize_attribute.argument_count());
            }

            no_attributes(&user_attributes, "specialize")?;

            SlangArgumentType::Specialize
        } else if let Some(axis_attribute) = user_attributes.remove("Axis") {
            if axis_attribute.argument_count() != 2 {
                bail!("axis expected 2 argument, found {}", axis_attribute.argument_count());
            }

            let threads = axis_attribute.argument_value_string(0).unwrap().to_string();
            let threads_in_group = axis_attribute.argument_value_string(1).unwrap().to_string();

            no_attributes(&user_attributes, "axis")?;

            SlangArgumentType::Axis {
                threads,
                threads_in_group,
            }
        } else if let Some(groups_attribute) = user_attributes.remove("Groups") {
            if groups_attribute.argument_count() != 1 {
                bail!("groups expected 1 argument, found {}", groups_attribute.argument_count());
            }

            let groups = groups_attribute.argument_value_string(0).unwrap().to_string();

            no_attributes(&user_attributes, "groups")?;

            SlangArgumentType::Groups {
                groups,
            }
        } else if let Some(threads_attribute) = user_attributes.remove("Threads") {
            if threads_attribute.argument_count() != 1 {
                bail!("threads expected 1 argument, found {}", threads_attribute.argument_count());
            }

            let threads = threads_attribute.argument_value_string(0).unwrap().to_string();

            no_attributes(&user_attributes, "threads")?;

            SlangArgumentType::Threads {
                threads,
            }
        } else if type_name == "ThreadContext" {
            no_attributes(&user_attributes, "thread context")?;

            SlangArgumentType::ThreadContext
        } else if type_name == "StructuredBuffer" {
            let condition = get_condition(&mut user_attributes)?;
            let is_constant = user_attributes.remove("Constant").is_some();

            no_attributes(&user_attributes, "buffer (read-only)")?;

            SlangArgumentType::Buffer {
                access_type: SlangBufferAccess::Read {
                    is_constant,
                },
                condition,
            }
        } else if type_name == "RWStructuredBuffer" {
            let condition = get_condition(&mut user_attributes)?;

            no_attributes(&user_attributes, "buffer (read-write)")?;

            SlangArgumentType::Buffer {
                access_type: SlangBufferAccess::ReadWrite,
                condition,
            }
        } else {
            let condition = get_condition(&mut user_attributes)?;

            no_attributes(&user_attributes, "constant")?;

            SlangArgumentType::Constant {
                condition,
            }
        };

        Ok(SlangArgument {
            name,
            slang_type: type_full_name,
            argument_type,
        })
    }

    pub fn rust_type(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Box<str>> {
        let slang_type = if matches!(
            self.argument_type,
            SlangArgumentType::Buffer {
                access_type: SlangBufferAccess::Read {
                    is_constant: true,
                },
                condition: _,
            }
        ) {
            let element_type = self.slang_type
                .strip_prefix("StructuredBuffer<")
                .and_then(|slang_type| slang_type.strip_suffix(", DefaultDataLayout>")).with_context(|| format!("buffer-constant argument doesn't match the StructuredBuffer<T, DefaultDataLayout> template: {}", self.slang_type))?;

            Cow::Owned(format!("{element_type}[]"))
        } else {
            Cow::Borrowed(&self.slang_type)
        };

        Ok(slang2rust(&slang_type, gpu_type_map)
            .with_context(|| format!("cannot convert slang {} to rust", self.slang_type))?
            .to_string()
            .into_boxed_str())
    }

    pub fn to_common_parameter(
        &self,
        gpu_type_map: &HashMap<String, String>,
    ) -> anyhow::Result<Option<KernelParameter>> {
        let SlangArgumentType::Specialize = self.argument_type else {
            return Ok(None);
        };

        Ok(Some(KernelParameter {
            name: self.name.clone().into_boxed_str(),
            ty: KernelParameterType::Value(self.rust_type(gpu_type_map)?),
        }))
    }

    pub fn to_common_argument(
        &self,
        gpu_type_map: &HashMap<String, String>,
        indirect_used: &mut bool,
    ) -> anyhow::Result<Option<KernelArgument>> {
        match &self.argument_type {
            SlangArgumentType::Buffer {
                access_type:
                    access_type @ (SlangBufferAccess::Read {
                        is_constant: false,
                    }
                    | SlangBufferAccess::ReadWrite),
                condition,
            } => Ok(Some(KernelArgument {
                name: self.name.clone().into_boxed_str(),
                conditional: condition.is_some(),
                ty: KernelArgumentType::Buffer(match *access_type {
                    SlangBufferAccess::Read {
                        is_constant: _,
                    } => KernelBufferAccess::Read,
                    SlangBufferAccess::ReadWrite => KernelBufferAccess::ReadWrite,
                }),
            })),
            SlangArgumentType::Constant {
                condition,
            }
            | SlangArgumentType::Buffer {
                access_type: SlangBufferAccess::Read {
                    is_constant: true,
                },
                condition,
            } => Ok(Some(KernelArgument {
                name: self.name.clone().into_boxed_str(),
                conditional: condition.is_some(),
                ty: KernelArgumentType::Constant(self.rust_type(gpu_type_map)?),
            })),
            SlangArgumentType::Groups {
                groups,
            } if !*indirect_used && groups == "INDIRECT" => {
                *indirect_used = true;

                Ok(Some(KernelArgument {
                    name: "__dsl_indirect_dispatch_buffer".into(),
                    conditional: false,
                    ty: KernelArgumentType::Buffer(KernelBufferAccess::Read),
                }))
            },
            _ => Ok(None),
        }
    }
}
