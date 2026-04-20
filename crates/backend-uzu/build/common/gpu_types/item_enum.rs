use anyhow::{Context, bail, ensure};
use syn::{Expr, ExprLit, ItemEnum, Lit, Variant};

use crate::common::gpu_types::ensure_repr_c;

#[derive(Debug)]
pub struct GpuTypeEnumVariant {
    pub name: Box<str>,
    pub discriminant: u32,
}

impl GpuTypeEnumVariant {
    fn parse(
        variant: Variant,
        discriminant: &mut u32,
    ) -> anyhow::Result<Self> {
        let name = variant.ident.to_string().into();

        if let Some((_, cur_discriminant)) = variant.discriminant {
            let Expr::Lit(ExprLit {
                attrs: _,
                lit: Lit::Int(cur_discriminant),
            }) = cur_discriminant
            else {
                bail!("Expected integer discriminant, found: {cur_discriminant:?}");
            };
            *discriminant = cur_discriminant.base10_parse().context("Cannot parse discriminant")?;
        }

        let cur_discriminant = *discriminant;
        *discriminant += 1;

        ensure!(variant.fields.is_empty(), "Enum variant with fields is not supported yet");

        Ok(Self {
            name,
            discriminant: cur_discriminant,
        })
    }
}

#[derive(Debug)]
pub struct GpuTypeEnum {
    pub name: Box<str>,
    pub variants: Box<[GpuTypeEnumVariant]>,
}

impl GpuTypeEnum {
    pub fn parse(item: ItemEnum) -> anyhow::Result<Self> {
        ensure_repr_c(&item.attrs)?;

        Ok(Self {
            name: item.ident.to_string().into(),
            variants: {
                let mut discriminant = 0;
                item.variants
                    .into_iter()
                    .map(|variant| GpuTypeEnumVariant::parse(variant, &mut discriminant))
                    .collect::<anyhow::Result<Box<[GpuTypeEnumVariant]>>>()?
            },
        })
    }
}
