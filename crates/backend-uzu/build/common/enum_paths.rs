use std::collections::HashMap;

use syn::{Type, TypePath, visit_mut::VisitMut};

use super::gpu_types::{GpuType, GpuTypes};

pub struct EnumPaths {
    short_name_to_full_path: HashMap<Box<str>, Box<str>>,
}

impl EnumPaths {
    pub fn from_gpu_types(gpu_types: &GpuTypes) -> Self {
        let mut short_name_to_full_path = HashMap::new();
        for file in gpu_types.files.iter() {
            for ty in file.types.iter() {
                if let GpuType::Enum(enum_type) = ty {
                    let full_path =
                        format!("crate::backends::common::gpu_types::{}::{}", file.name, enum_type.name)
                            .into_boxed_str();
                    short_name_to_full_path.insert(enum_type.name.clone(), full_path);
                }
            }
        }
        Self {
            short_name_to_full_path,
        }
    }

    pub fn full_path_for(
        &self,
        short_name: &str,
    ) -> Option<&str> {
        self.short_name_to_full_path.get(short_name).map(|path| path.as_ref())
    }

    pub fn contains(
        &self,
        short_name: &str,
    ) -> bool {
        self.short_name_to_full_path.contains_key(short_name)
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

struct TypePathCanonicalizer<'enum_paths> {
    enum_paths: &'enum_paths EnumPaths,
}

impl<'enum_paths> VisitMut for TypePathCanonicalizer<'enum_paths> {
    fn visit_type_path_mut(
        &mut self,
        type_path: &mut TypePath,
    ) {
        let path = &type_path.path;
        if type_path.qself.is_none()
            && path.leading_colon.is_none()
            && path.segments.len() == 1
            && matches!(path.segments[0].arguments, syn::PathArguments::None)
        {
            let segment_name = path.segments[0].ident.to_string();
            if let Some(full_path_text) = self.enum_paths.full_path_for(&segment_name)
                && let Ok(full_path) = syn::parse_str::<syn::Path>(full_path_text)
            {
                type_path.path = full_path;
                return;
            }
        }
        syn::visit_mut::visit_type_path_mut(self, type_path);
    }
}
