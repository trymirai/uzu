#![allow(unused)]
use std::{
    borrow::Borrow,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use derive_more::{AsRef, Deref, Display, From};
use syn::{Attribute, Ident, Item};
use walkdir::WalkDir;

mod item_enum;
mod item_option_set;
mod item_struct;
mod item_variant_group;

pub use item_enum::GpuTypeEnum;
pub use item_option_set::{GpuTypeOptionSet, GpuTypeOptionSetVariant};
pub use item_struct::{GpuTypeStruct, GpuTypeStructFieldType};
pub use item_variant_group::{GpuTypeVariantGroup, VariantGroupArm};

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
    fn scan(path: &Path) -> anyhow::Result<Self> {
        let name = path.file_stem().context("No file stem")?.to_string_lossy().into();

        let source = fs::read_to_string(path).context("Cannot read source")?;
        let ast = syn::parse_file(&source).context("Cannot parse ast")?;

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
            name,
            types: tys,
        })
    }
}

#[derive(Debug)]
pub struct GpuTypes {
    pub files: Box<[GpuTypeFile]>,
}

impl GpuTypes {
    pub fn scan() -> anyhow::Result<Self> {
        let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("Missing CARGO_MANIFEST_DIR")?)
            .join("src/backends/common/gpu_types");

        let mut sources: Vec<PathBuf> = WalkDir::new(&src_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_type().is_file().then(|| e.into_path()))
            .filter(|e| e.extension().and_then(|s| s.to_str()) == Some("rs") && e.file_stem() != Some("mod".as_ref()))
            .collect();

        sources.sort();

        Ok(Self {
            files: sources
                .into_iter()
                .map(|source| {
                    GpuTypeFile::scan(&source)
                        .with_context(|| format!("Failed to scan {}", source.as_os_str().to_str().unwrap()))
                })
                .collect::<anyhow::Result<Box<[GpuTypeFile]>>>()?,
        })
    }
}
