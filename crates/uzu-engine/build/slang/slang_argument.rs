use anyhow::{Context, bail, ensure};
use quote::ToTokens;
use shader_slang::{
    ScalarType, TypeKind,
    reflection::{Generic, UserAttribute, Variable},
};

use super::{Error, SlangArgumentType};
use crate::common::{
    gpu_type_paths::{GpuTypeKind, GpuTypePaths},
    kernel::KernelBufferAccess,
};

pub struct SlangArgument<'a> {
    variable: &'a Variable,
    gpu_type_paths: &'a GpuTypePaths,
}

impl<'a> SlangArgument<'a> {
    pub fn new(
        variable: &'a Variable,
        gpu_type_paths: &'a GpuTypePaths,
    ) -> Self {
        Self {
            variable,
            gpu_type_paths,
        }
    }

    pub fn name(&self) -> Result<&str, Error> {
        self.variable.name().context("Slang argument has no name")
    }

    pub fn slang_type(&self) -> Result<String, Error> {
        let ty = self.variable.ty().context("Slang argument has no type")?;
        Ok(ty.full_name()?.as_str()?.to_string())
    }

    pub fn specialized_slang_type(
        &self,
        generic: &Generic,
    ) -> Result<String, Error> {
        let ty = self.variable.ty().context("Slang argument has no type")?;
        let specialized = ty.apply_specializations(generic).context("cannot specialize Slang argument type")?;
        Ok(specialized.full_name()?.as_str()?.to_string())
    }

    /// Host condition under which an `[[Optional]]` argument is present.
    pub fn condition(&self) -> Result<Option<&'a str>, Error> {
        self.attribute("Optional")
            .map(|optional| optional.argument_value_string(0).context("Optional missing condition"))
            .transpose()
    }

    /// Constants and specializations carry their canonical Rust type; an optional specialization is an `Option` of it.
    pub fn argument_type(&self) -> Result<SlangArgumentType, Error> {
        let ty = self.variable.ty().context("Slang argument has no type")?;
        let value = match ty.kind() {
            TypeKind::Scalar => match ty.scalar_type() {
                ScalarType::Bool => Some("bool".to_string()),
                ScalarType::Uint32 => Some("u32".to_string()),
                ScalarType::Int32 => Some("i32".to_string()),
                ScalarType::Float32 => Some("f32".to_string()),
                _ => None,
            },
            TypeKind::Enum => {
                let name = self.slang_type()?;
                ensure!(
                    self.gpu_type_paths.kind_for(&name) == Some(GpuTypeKind::Enum),
                    "'{}' has enum type '{name}' that is not a canonical GPU enum",
                    self.name()?
                );
                Some(name)
            },
            // A by-value canonical struct, passed like the CPU kernel's value of it.
            TypeKind::Struct => {
                let name = self.slang_type()?;
                ensure!(
                    self.gpu_type_paths.kind_for(&name) == Some(GpuTypeKind::Struct),
                    "'{}' has struct type '{name}' that is not a canonical GPU struct",
                    self.name()?
                );
                ensure!(self.attribute("Specialize").is_none(), "'{}': a struct cannot be specialized", self.name()?);
                Some(name)
            },
            _ => None,
        };

        // A host slice is the CPU kernel's `&[T]` of a canonical struct, whose pointer the shader reads.
        if self.attribute("HostSlice").is_some() {
            let name = self.name()?;
            if let Some(other) = ["Optional", "Specialize", "Axis", "Groups", "Threads", "PipelineVariants"]
                .into_iter()
                .find(|other| self.attribute(other).is_some())
            {
                bail!("'{name}': HostSlice cannot be combined with {other}");
            }
            ensure!(matches!(ty.kind(), TypeKind::Pointer), "'{name}': HostSlice needs a pointer");
            let (pointee, access) = self.pointer()?;
            ensure!(access == KernelBufferAccess::Read, "'{name}': HostSlice needs a read-only pointer");
            ensure!(
                self.gpu_type_paths.kind_for(&pointee) == Some(GpuTypeKind::Struct),
                "'{name}': HostSlice points to '{pointee}', which is not a canonical GPU struct"
            );
            return Ok(SlangArgumentType::Constant(format!("&[{}]", self.rust_type(&pointee)?).into()));
        }
        if let Some(axis) = self.attribute("Axis") {
            let total = axis.argument_value_string(0).context("Axis missing arg 0")?.into();
            let per_group = axis.argument_value_string(1).context("Axis missing arg 1")?.into();
            Ok(SlangArgumentType::Axis(total, per_group))
        } else if let Some(groups) = self.attribute("Groups") {
            Ok(SlangArgumentType::Groups(groups.argument_value_string(0).context("Groups missing arg")?.into()))
        } else if let Some(threads) = self.attribute("Threads") {
            Ok(SlangArgumentType::Threads(threads.argument_value_string(0).context("Threads missing arg")?.into()))
        } else if self.attribute("Specialize").is_some() {
            match value {
                Some(value) if value == "bool" && self.condition()?.is_none() => {
                    Ok(SlangArgumentType::Specialize(value.into()))
                },
                Some(value) if !matches!(value.as_str(), "bool" | "i32" | "f32") => {
                    Ok(SlangArgumentType::Specialize(match self.condition()? {
                        Some(_) => self.rust_type(&format!("Option<{value}>"))?,
                        None => self.rust_type(&value)?,
                    }))
                },
                _ => bail!("unsupported specialization type for '{}': {}", self.name()?, self.slang_type()?),
            }
        } else {
            match (ty.kind(), value) {
                (TypeKind::Pointer, _) => Ok(SlangArgumentType::Ptr(self.pointer()?.1)),
                (_, Some(value)) => Ok(SlangArgumentType::Constant(self.rust_type(&value)?)),
                (kind, _) => bail!(
                    "unsupported parameter type for '{}': kind={kind:?} name={}",
                    self.name()?,
                    self.slang_type()?
                ),
            }
        }
    }

    /// Whether the argument is `[[PipelineVariants]]`, a uniform canonical GPU enum without other annotations.
    pub fn pipeline_variants(&self) -> Result<bool, Error> {
        if self.attribute("PipelineVariants").is_none() {
            return Ok(false);
        }
        let name = self.name()?;
        if let Some(other) = ["Optional", "Specialize", "Axis", "Groups", "Threads"]
            .into_iter()
            .find(|other| self.attribute(other).is_some())
        {
            bail!("'{name}': PipelineVariants cannot be combined with {other}");
        }
        let ty = self.variable.ty().context("Slang argument has no type")?;
        ensure!(
            matches!(ty.kind(), TypeKind::Enum) && matches!(self.argument_type()?, SlangArgumentType::Constant(_)),
            "'{name}': PipelineVariants needs a uniform canonical GPU enum, not {}",
            self.slang_type()?
        );
        Ok(true)
    }

    /// The type text of the common kernel descriptor: enum names resolve to their canonical paths.
    fn rust_type(
        &self,
        text: &str,
    ) -> Result<Box<str>, Error> {
        let ty = self.gpu_type_paths.resolve_type(text)?;
        Ok(ty.to_token_stream().to_string().replace(" :: ", "::").into())
    }

    fn attribute(
        &self,
        name: &str,
    ) -> Option<&'a UserAttribute> {
        self.variable.user_attributes().find(|attribute| attribute.name() == Some(name))
    }

    /// Pointee and access mode of a reflected `Ptr<T, Access.X, AddressSpace.Device[, DefaultDataLayout]>`.
    fn pointer(&self) -> Result<(String, KernelBufferAccess), Error> {
        let ty = self.slang_type()?;
        let pointer = ty
            .strip_prefix("Ptr<")
            .and_then(|ty| {
                ty.strip_suffix(", AddressSpace.Device>")
                    .or_else(|| ty.strip_suffix(", AddressSpace.Device, DefaultDataLayout>"))
            })
            .and_then(|ty| ty.rsplit_once(", Access."));
        match pointer {
            Some((pointee, "Read")) => Ok((pointee.to_owned(), KernelBufferAccess::Read)),
            Some((pointee, "ReadWrite")) => Ok((pointee.to_owned(), KernelBufferAccess::ReadWrite)),
            _ => bail!("unsupported pointer type for '{}': {ty}", self.name()?),
        }
    }
}
