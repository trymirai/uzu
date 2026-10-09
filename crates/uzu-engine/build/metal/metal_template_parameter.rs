use super::{Error, ast::MetalTemplateParameterType};
use crate::common::{KernelParameterType, data_type, kernel::KernelParameter};

#[derive(Debug)]
pub struct MetalTemplateParameter {
    pub name: Box<str>,
    pub ty: MetalTemplateParameterType,
    pub variants: Box<[Box<str>]>,
}

impl MetalTemplateParameter {
    pub fn to_parameter(&self) -> Result<KernelParameter, Error> {
        Ok(KernelParameter {
            name: self.name.clone(),
            ty: match &self.ty {
                MetalTemplateParameterType::Type => {
                    let data_types = self.variants.iter().map(|variant| Ok(data_type(variant)?.to_string().into()));
                    KernelParameterType::types(data_types.collect::<Result<Vec<_>, Error>>()?)
                },
                MetalTemplateParameterType::Value(ty) => KernelParameterType::Value(ty.clone()),
            },
        })
    }
}
