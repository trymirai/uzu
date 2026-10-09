use std::collections::HashMap;

use anyhow::{Context, bail};
use shader_slang::{
    ScalarType, TypeKind,
    reflection::{Generic, UserAttribute, Variable},
};

use super::{Error, SlangArgumentType};

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

    pub fn argument_type(&self) -> Result<SlangArgumentType, Error> {
        let ty = self.variable.ty().context("Slang argument has no type")?;
        let attrs = self
            .variable
            .user_attributes()
            .map(|attribute| Ok((attribute.name().context("Slang attribute has no name")?, attribute)))
            .collect::<Result<HashMap<&str, &UserAttribute>, Error>>()?;

        if let Some(axis) = attrs.get("Axis") {
            let total = axis.argument_value_string(0).context("Axis missing arg 0")?.into();
            let per_group = axis.argument_value_string(1).context("Axis missing arg 1")?.into();
            Ok(SlangArgumentType::Axis(total, per_group))
        } else if let Some(groups) = attrs.get("Groups") {
            groups.argument_value_string(0).context("Groups missing arg")?;
            Ok(SlangArgumentType::Groups)
        } else if let Some(threads) = attrs.get("Threads") {
            Ok(SlangArgumentType::Threads(threads.argument_value_string(0).context("Threads missing arg")?.into()))
        } else {
            match ty.kind() {
                TypeKind::Pointer => Ok(SlangArgumentType::Ptr),
                TypeKind::Scalar => {
                    match ty.scalar_type() {
                        ScalarType::Uint32 | ScalarType::Int32 | ScalarType::Float32 | ScalarType::Float16 => {},
                        other => bail!("unsupported scalar type: {other:?}"),
                    }
                    Ok(SlangArgumentType::Constant)
                },
                other => bail!(
                    "unsupported parameter type for '{}': kind={other:?} name={}",
                    self.name()?,
                    self.slang_type()?
                ),
            }
        }
    }
}
