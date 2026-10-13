use std::collections::{HashMap, hash_map::Entry};

use anyhow::{Context, bail};
use syn::{Type, TypePath, visit_mut::VisitMut};

use crate::common::gpu_types::{GpuType, GpuTypeName, GpuTypePath, GpuTypes};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuTypeKind {
    Enum,
    OptionSet,
    Struct,
}

#[derive(Clone)]
struct GpuTypeEntry {
    path: GpuTypePath,
    kind: GpuTypeKind,
}

#[derive(Clone)]
pub struct GpuTypePaths {
    short_name_to_entry: HashMap<GpuTypeName, GpuTypeEntry>,
}

impl GpuTypePaths {
    pub fn from_gpu_types(gpu_types: &GpuTypes) -> anyhow::Result<Self> {
        let mut paths = Self {
            short_name_to_entry: HashMap::new(),
        };
        paths.extend(gpu_types)?;
        Ok(paths)
    }

    pub fn extend(
        &mut self,
        gpu_types: &GpuTypes,
    ) -> anyhow::Result<()> {
        for file in gpu_types.files.iter() {
            for ty in file.types.iter() {
                let (name_str, kind) = match ty {
                    GpuType::Enum(enum_type) => (enum_type.name.as_ref(), GpuTypeKind::Enum),
                    GpuType::OptionSet(option_set) => (option_set.name.as_ref(), GpuTypeKind::OptionSet),
                    GpuType::Struct(struct_type) => (struct_type.name.as_ref(), GpuTypeKind::Struct),
                    GpuType::Constant(_) => continue,
                };
                let name = GpuTypeName::from(name_str);
                let path = GpuTypePath::from(format!("{}::{}::{}", gpu_types.module_path, file.name, name_str));
                match self.short_name_to_entry.entry(name) {
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
        Ok(())
    }

    pub fn full_path_for(
        &self,
        short_name: &str,
    ) -> Option<&str> {
        self.short_name_to_entry.get(short_name).map(|entry| &*entry.path)
    }

    #[allow(dead_code)]
    pub fn full_paths(&self) -> impl Iterator<Item = &str> {
        self.short_name_to_entry.values().map(|entry| &*entry.path)
    }

    #[allow(dead_code)]
    pub fn kind_for(
        &self,
        short_name: &str,
    ) -> Option<GpuTypeKind> {
        self.short_name_to_entry.get(short_name).map(|entry| entry.kind)
    }

    pub fn resolve_type(
        &self,
        type_text: &str,
    ) -> anyhow::Result<Type> {
        struct TypePaths<'a>(&'a GpuTypePaths);

        impl VisitMut for TypePaths<'_> {
            fn visit_type_path_mut(
                &mut self,
                ty: &mut TypePath,
            ) {
                syn::visit_mut::visit_type_path_mut(self, ty);
                if ty.qself.is_none()
                    && let Some(ident) = ty.path.get_ident()
                    && let Some(path) = self.0.full_path_for(&ident.to_string())
                {
                    ty.path = syn::parse_str(path).unwrap();
                }
            }
        }

        let mut ty = syn::parse_str(type_text).with_context(|| format!("cannot parse type `{type_text}`"))?;
        TypePaths(self).visit_type_mut(&mut ty);
        Ok(ty)
    }
}
