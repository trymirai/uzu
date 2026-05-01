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
        attrs.iter().any(|attr| {
            if !attr.path().is_ident("repr") {
                return false;
            }
            attr.parse_args_with(syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
                .map(|metas| metas.iter().any(|m| m.path().is_ident("C")))
                .unwrap_or(false)
        }),
        "Does not have repr(c)"
    );

    Ok(())
}

pub fn parse_repr_alignment(attrs: &[Attribute]) -> Option<u32> {
    for attr in attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }
        let Ok(metas) =
            attr.parse_args_with(syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
        else {
            continue;
        };
        for meta in metas {
            if let syn::Meta::List(list) = &meta
                && list.path.is_ident("align")
                && let Ok(lit) = list.parse_args::<syn::LitInt>()
                && let Ok(n) = lit.base10_parse::<u32>()
            {
                return Some(n);
            }
        }
    }
    None
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

fn parse_types_from_file(path: &Path) -> anyhow::Result<Vec<GpuType>> {
    let source = fs::read_to_string(path).context("Cannot read source")?;
    let ast = syn::parse_file(&source).context("Cannot parse ast")?;

    ast.items
        .into_iter()
        .filter_map(|item| match item {
            Item::Enum(item) => Some(GpuTypeEnum::parse(item).map(GpuType::Enum)),
            Item::Struct(item) => Some(GpuTypeStruct::parse(item).map(GpuType::Struct)),
            _ => None,
        })
        .collect()
}

impl GpuTypeFile {
    fn scan_file(path: &Path) -> anyhow::Result<Self> {
        let name = path.file_stem().context("No file stem")?.to_string_lossy().into();
        let types = parse_types_from_file(path)?.into_boxed_slice();
        Ok(Self {
            name,
            types,
        })
    }

    fn scan_dir(path: &Path) -> anyhow::Result<Self> {
        let name = path.file_name().context("No directory name")?.to_string_lossy().into();

        let mut sources: Vec<PathBuf> = WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_type().is_file().then(|| e.into_path()))
            .filter(|e| e.extension().and_then(|s| s.to_str()) == Some("rs"))
            .collect();
        sources.sort();

        let mut all_types: Vec<GpuType> = Vec::new();
        for source in sources {
            let mut types = parse_types_from_file(&source)
                .with_context(|| format!("Failed to scan {}", source.display()))?;
            all_types.append(&mut types);
        }

        Ok(Self {
            name,
            types: all_types.into_boxed_slice(),
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

        let mut entries: Vec<PathBuf> = fs::read_dir(&src_dir)
            .with_context(|| format!("Cannot read {}", src_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                if p.is_file() {
                    p.extension().and_then(|s| s.to_str()) == Some("rs")
                        && p.file_stem().and_then(|s| s.to_str()) != Some("mod")
                } else {
                    p.is_dir()
                }
            })
            .collect();

        entries.sort();

        let files = entries
            .into_iter()
            .map(|entry| {
                if entry.is_dir() {
                    GpuTypeFile::scan_dir(&entry)
                        .with_context(|| format!("Failed to scan dir {}", entry.display()))
                } else {
                    GpuTypeFile::scan_file(&entry)
                        .with_context(|| format!("Failed to scan {}", entry.display()))
                }
            })
            .collect::<anyhow::Result<Box<[GpuTypeFile]>>>()?;

        Ok(Self {
            files,
        })
    }
}
