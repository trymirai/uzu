use anyhow::{Context, anyhow, bail};
use quote::ToTokens;
use syn::{Expr, ExprLit, Fields, ItemStruct, Lit, Type, TypeArray, TypePath};

use crate::common::gpu_types::{ensure_repr_c, parse_repr_alignment};

#[derive(Debug)]
pub enum GpuTypeStructFieldType {
    Scalar(Box<str>),
    Array {
        element: Box<str>,
        length: usize,
    },
}

impl GpuTypeStructFieldType {
    fn parse(ty: Type) -> anyhow::Result<Self> {
        match ty {
            Type::Path(TypePath {
                qself: _,
                path,
            }) => Ok(Self::Scalar(path.into_token_stream().to_string().into())),
            Type::Array(TypeArray {
                bracket_token: _,
                elem,
                semi_token: _,
                len,
            }) => {
                let Type::Path(elem) = *elem else {
                    bail!("Array element must be a Path, found {elem:?}");
                };

                let Expr::Lit(ExprLit {
                    attrs: _,
                    lit: Lit::Int(len),
                }) = len
                else {
                    bail!("Array length must be an integer literal, found: {len:?}");
                };

                Ok(Self::Array {
                    element: elem.into_token_stream().to_string().into(),
                    length: len.base10_parse().context("Cannot parse array length")?,
                })
            },
            ty => Err(anyhow!("Expected Array or Path, found {ty:?}")),
        }
    }
}

#[derive(Debug)]
pub struct GpuTypeStructField {
    pub name: Box<str>,
    pub ty: GpuTypeStructFieldType,
}

#[derive(Debug)]
pub struct GpuTypeStruct {
    pub name: Box<str>,
    pub fields: Box<[GpuTypeStructField]>,
    pub alignment: Option<u32>,
}

impl GpuTypeStruct {
    pub fn parse(item: ItemStruct) -> anyhow::Result<Self> {
        ensure_repr_c(&item.attrs)?;
        let alignment = parse_repr_alignment(&item.attrs);

        let name = item.ident.to_string().into();

        let Fields::Named(item_fields) = item.fields else {
            bail!("Only structs with named fields are supported, found: {:?}", item.fields);
        };

        let fields = item_fields
            .named
            .into_iter()
            .map(|field| {
                Ok(GpuTypeStructField {
                    name: field.ident.context("Field doesn't have ident")?.to_string().into(),
                    ty: GpuTypeStructFieldType::parse(field.ty).context("Cannot parse field type")?,
                })
            })
            .collect::<anyhow::Result<Box<[GpuTypeStructField]>>>()?;

        Ok(Self {
            name,
            fields,
            alignment,
        })
    }
}
