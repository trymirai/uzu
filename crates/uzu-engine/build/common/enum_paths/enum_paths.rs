use std::collections::{HashMap, hash_map::Entry};

use anyhow::bail;
use syn::{Type, visit_mut::VisitMut};

use super::{GpuTypeEntry, GpuTypeKind, TypePathCanonicalizer};
use crate::common::gpu_types::{GpuType, GpuTypeName, GpuTypePath, GpuTypes};

/// Canonical paths of the GPU enums, option sets and structs, by short name.
#[derive(Clone)]
pub struct EnumPaths {
    short_name_to_entry: HashMap<GpuTypeName, GpuTypeEntry>,
}

impl EnumPaths {
    pub fn from_gpu_types(gpu_types: &GpuTypes) -> anyhow::Result<Self> {
        let mut short_name_to_entry = HashMap::new();
        for file in gpu_types.files.iter() {
            for ty in file.types.iter() {
                let (name_str, kind) = match ty {
                    GpuType::Enum(enum_type) => (enum_type.name.as_ref(), Some(GpuTypeKind::Enum)),
                    GpuType::OptionSet(option_set) => (option_set.name.as_ref(), Some(GpuTypeKind::OptionSet)),
                    GpuType::Struct(struct_type) => (struct_type.name.as_ref(), None),
                    GpuType::Constant(_) => continue,
                };
                let name = GpuTypeName::from(name_str);
                let path =
                    GpuTypePath::from(format!("crate::backends::common::gpu_types::{}::{}", file.name, name_str));
                match short_name_to_entry.entry(name) {
                    Entry::Occupied(occupied) => {
                        bail!("gpu type `{}` is duplicated", occupied.key())
                    },
                    Entry::Vacant(vacant) => {
                        vacant.insert(GpuTypeEntry {
                            path,
                            kind,
                        });
                    },
                }
            }
        }
        Ok(Self {
            short_name_to_entry,
        })
    }

    pub fn full_path_for(
        &self,
        short_name: &str,
    ) -> Option<&str> {
        self.short_name_to_entry.get(short_name).map(|entry| &*entry.path)
    }

    /// The scalar kind of a canonical enum or option set; `None` for canonical structs, which `full_path_for` knows,
    /// and unknown names.
    #[allow(dead_code)]
    pub fn kind_for(
        &self,
        short_name: &str,
    ) -> Option<GpuTypeKind> {
        self.short_name_to_entry.get(short_name).and_then(|entry| entry.kind)
    }

    pub fn canonicalize_type(
        &self,
        ty: &mut Type,
    ) {
        let mut canonicalizer = TypePathCanonicalizer {
            enum_paths: self,
        };
        canonicalizer.visit_type_mut(ty);
    }
}
