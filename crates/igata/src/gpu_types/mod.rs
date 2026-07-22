use std::borrow::Borrow;

use anyhow::Context;
use derive_more::{AsRef, Deref, Display, From};
use syn::{Attribute, Ident, Item};

mod item_enum;
mod item_option_set;
mod item_struct;
mod item_variant_group;
pub mod tile_geometry;
mod tile_geometry_gen;

pub use item_enum::GpuTypeEnum;
pub use item_option_set::{GpuTypeOptionSet, GpuTypeOptionSetVariant};
pub use item_struct::{GpuTypeStruct, GpuTypeStructFieldType};
pub use item_variant_group::{GpuTypeVariantGroup, VariantGroupArm};
pub use tile_geometry_gen::tile_geometry_tokens;

#[derive(Clone, Debug, PartialEq, Eq, Hash, From, AsRef, Deref, Display)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct GpuTypeName(String);

impl Borrow<str> for GpuTypeName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, From, AsRef, Deref, Display)]
#[as_ref(str)]
#[deref(forward)]
#[from(forward)]
pub struct GpuTypePath(String);

fn ensure_repr_c(attrs: &[Attribute]) -> anyhow::Result<()> {
    anyhow::ensure!(
        attrs.iter().any(|attr| attr.path().is_ident("repr") && attr.parse_args::<Ident>().is_ok_and(|arg| arg == "C")),
        "Does not have repr(c)"
    );

    Ok(())
}

#[derive(Debug)]
pub enum GpuType {
    Enum(GpuTypeEnum),
    Struct(GpuTypeStruct),
    OptionSet(GpuTypeOptionSet),
    /// A Rust-side key type, not emitted into the Metal header.
    VariantGroup(GpuTypeVariantGroup),
}

#[derive(Debug)]
pub struct GpuTypeFile {
    pub name: Box<str>,
    pub types: Box<[GpuType]>,
}

impl GpuTypeFile {
    /// Parses one `gpu_types` module: `name` is its file stem, which the generated Rust
    /// and Metal paths are built from.
    pub fn parse(
        name: &str,
        source: &str,
    ) -> anyhow::Result<Self> {
        let ast = syn::parse_file(source).context("Cannot parse ast")?;

        let tys = ast
            .items
            .into_iter()
            .filter_map(|item| match item {
                Item::Enum(item) => Some(match GpuTypeVariantGroup::axes_of(&item.attrs) {
                    Ok(Some(axes)) => GpuTypeVariantGroup::parse(item, axes).map(GpuType::VariantGroup),
                    Ok(None) => GpuTypeEnum::parse(item).map(GpuType::Enum),
                    Err(error) => Err(error),
                }),
                Item::Struct(item) => Some(GpuTypeStruct::parse(item).map(GpuType::Struct)),
                Item::Macro(item) if item.mac.path.is_ident("bitflags") => {
                    Some(GpuTypeOptionSet::parse(item.mac.tokens).map(GpuType::OptionSet))
                },
                _ => None,
            })
            .collect::<anyhow::Result<Box<[GpuType]>>>()?;

        Ok(Self {
            name: name.into(),
            types: tys,
        })
    }
}

#[derive(Debug)]
pub struct GpuTypes {
    pub files: Box<[GpuTypeFile]>,
}

impl GpuTypes {
    pub fn new(files: impl IntoIterator<Item = GpuTypeFile>) -> Self {
        Self {
            files: files.into_iter().collect(),
        }
    }
}
