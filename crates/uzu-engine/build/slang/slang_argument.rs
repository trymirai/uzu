use anyhow::{Context, bail};
use shader_slang::{
    ScalarType, TypeKind,
    reflection::{Generic, UserAttribute, Variable},
};

use super::{Error, SlangArgumentType};
use crate::common::kernel::KernelBufferAccess;

pub struct SlangArgument<'a> {
    variable: &'a Variable,
}

impl<'a> SlangArgument<'a> {
    pub fn new(variable: &'a Variable) -> Self {
        Self {
            variable,
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

    pub fn argument_type(&self) -> Result<SlangArgumentType, Error> {
        let ty = self.variable.ty().context("Slang argument has no type")?;
        let scalar = matches!(ty.kind(), TypeKind::Scalar).then(|| ty.scalar_type());

        if let Some(axis) = self.attribute("Axis") {
            let total = axis.argument_value_string(0).context("Axis missing arg 0")?.into();
            let per_group = axis.argument_value_string(1).context("Axis missing arg 1")?.into();
            Ok(SlangArgumentType::Axis(total, per_group))
        } else if let Some(groups) = self.attribute("Groups") {
            groups.argument_value_string(0).context("Groups missing arg")?;
            Ok(SlangArgumentType::Groups)
        } else if let Some(threads) = self.attribute("Threads") {
            Ok(SlangArgumentType::Threads(threads.argument_value_string(0).context("Threads missing arg")?.into()))
        } else if self.attribute("Specialize").is_some() {
            match scalar {
                Some(ScalarType::Bool) => Ok(SlangArgumentType::Specialize("bool".into())),
                _ => bail!("unsupported specialization type for '{}': {}", self.name()?, self.slang_type()?),
            }
        } else {
            match (ty.kind(), scalar) {
                (TypeKind::Pointer, _) => Ok(SlangArgumentType::Ptr(self.access()?)),
                (_, Some(ScalarType::Uint32)) => Ok(SlangArgumentType::Constant("u32".into())),
                (_, Some(ScalarType::Int32)) => Ok(SlangArgumentType::Constant("i32".into())),
                (_, Some(ScalarType::Float32)) => Ok(SlangArgumentType::Constant("f32".into())),
                (kind, _) => bail!(
                    "unsupported parameter type for '{}': kind={kind:?} name={}",
                    self.name()?,
                    self.slang_type()?
                ),
            }
        }
    }

    fn attribute(
        &self,
        name: &str,
    ) -> Option<&'a UserAttribute> {
        self.variable.user_attributes().find(|attribute| attribute.name() == Some(name))
    }

    /// Access mode as reflected by Slang, e.g. `Ptr<T, Access.Read, AddressSpace.Device>`.
    fn access(&self) -> Result<KernelBufferAccess, Error> {
        let ty = self.slang_type()?;
        let access = ty
            .strip_prefix("Ptr<")
            .and_then(|ty| ty.strip_suffix(", AddressSpace.Device>"))
            .and_then(|ty| ty.rsplit_once(", Access."))
            .map(|(_, access)| access);
        match access {
            Some("Read") => Ok(KernelBufferAccess::Read),
            Some("ReadWrite") => Ok(KernelBufferAccess::ReadWrite),
            _ => bail!("unsupported pointer type for '{}': {ty}", self.name()?),
        }
    }
}
