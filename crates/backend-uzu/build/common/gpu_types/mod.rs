#![allow(unused)]
use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use syn::{Attribute, Ident, Item};
use walkdir::WalkDir;

mod item_enum;
mod item_struct;

pub use item_enum::GpuTypeEnum;
pub use item_struct::{GpuTypeStruct, GpuTypeStructFieldType};

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
                Item::Enum(item) => Some(GpuTypeEnum::parse(item).map(GpuType::Enum)),
                Item::Struct(item) => Some(GpuTypeStruct::parse(item).map(GpuType::Struct)),
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
