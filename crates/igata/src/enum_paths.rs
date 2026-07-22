use std::collections::{BTreeMap, HashMap, hash_map::Entry};

use anyhow::bail;
use syn::{Type, TypePath, visit_mut::VisitMut};

use crate::{
    constraint_expr,
    gpu_types::{
        GpuType, GpuTypeName, GpuTypePath, GpuTypeVariantGroup, GpuTypes, VariantGroupArm,
        tile_geometry::{ACCESSORS, geometries},
    },
    mangling::snake_case,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuTypeKind {
    Enum,
    OptionSet,
}

/// Length-prefixed so that concatenating fields cannot alias a different field split.
fn hash_str(
    hasher: &mut blake3::Hasher,
    text: &str,
) {
    hasher.update(&(text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

#[derive(Clone)]
struct GpuTypeEntry {
    path: GpuTypePath,
    kind: GpuTypeKind,
    /// Variant names and discriminants in declaration order, so a shader can omit a
    /// `VARIANTS` list for an enum axis and have the members single-sourced from the
    /// Rust definition.
    variants: Box<[(Box<str>, u32)]>,
}

#[derive(Clone)]
pub struct EnumPaths {
    short_name_to_entry: HashMap<GpuTypeName, GpuTypeEntry>,
    variant_groups: Box<[GpuTypeVariantGroup]>,
    helpers: constraint_expr::Helpers,
    variant_group_paths: BTreeMap<Box<str>, Box<str>>,
}

impl EnumPaths {
    pub fn from_gpu_types(gpu_types: &GpuTypes) -> anyhow::Result<Self> {
        let mut short_name_to_entry = HashMap::new();
        let mut variant_groups = Vec::new();
        let mut helpers = constraint_expr::Helpers::new();
        let mut variant_group_paths = BTreeMap::new();
        for file in gpu_types.files.iter() {
            for ty in file.types.iter() {
                let (name_str, kind, variants) = match ty {
                    GpuType::Enum(enum_type) => (
                        {
                            if let Some(tiles) = geometries(enum_type) {
                                let prefix = snake_case(&enum_type.name);
                                for (_, metal_suffix, value_of) in ACCESSORS {
                                    helpers.insert(
                                        format!("{prefix}_{metal_suffix}").into_boxed_str(),
                                        constraint_expr::Helper {
                                            parameter: constraint_expr::Type::Enum(enum_type.name.clone()),
                                            values: tiles
                                                .iter()
                                                .map(|(variant, geometry)| {
                                                    ((*variant).into(), i64::from(value_of(geometry)))
                                                })
                                                .collect(),
                                        },
                                    );
                                }
                            }
                            enum_type.name.as_ref()
                        },
                        GpuTypeKind::Enum,
                        enum_type.variants.iter().map(|v| (v.name.clone(), v.discriminant)).collect(),
                    ),
                    GpuType::OptionSet(option_set) => {
                        (option_set.name.as_ref(), GpuTypeKind::OptionSet, Box::default())
                    },
                    GpuType::Struct(_) => continue,
                    GpuType::VariantGroup(group) => {
                        variant_group_paths.insert(
                            group.name.clone(),
                            format!("crate::backends::common::gpu_types::{}::{}", file.name, group.name)
                                .into_boxed_str(),
                        );
                        variant_groups.push(group.clone());
                        continue;
                    },
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
                            variants,
                        });
                    },
                }
            }
        }
        Ok(Self {
            short_name_to_entry,
            variant_groups: variant_groups.into(),
            helpers,
            variant_group_paths,
        })
    }

    /// Everything the generators read out of the Rust `gpu_types` sources, hashed.
    ///
    /// Wrappers, bindings and the manifest are all derived from this data, but a variant
    /// group emits nothing into the Metal headers whose contents the object cache hashes:
    /// editing one leaves every other cache input identical, so the caches would serve an
    /// `.air` and a `dsl.rs` built for a variant set that no longer exists. Mixing this
    /// in is what makes such an edit visible to them.
    pub fn semantic_fingerprint(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();

        hash_str(&mut hasher, "enums");
        let entries: BTreeMap<&str, &GpuTypeEntry> =
            self.short_name_to_entry.iter().map(|(name, entry)| (name.as_ref(), entry)).collect();
        for (name, entry) in entries {
            hash_str(&mut hasher, name);
            hash_str(&mut hasher, &entry.path);
            hasher.update(&[match entry.kind {
                GpuTypeKind::Enum => 0,
                GpuTypeKind::OptionSet => 1,
            }]);
            for (variant, discriminant) in entry.variants.iter() {
                hash_str(&mut hasher, variant);
                hasher.update(&discriminant.to_le_bytes());
            }
        }

        hash_str(&mut hasher, "variant groups");
        let groups: BTreeMap<&str, &GpuTypeVariantGroup> =
            self.variant_groups.iter().map(|group| (group.name.as_ref(), group)).collect();
        for (name, group) in groups {
            hash_str(&mut hasher, name);
            for axis in group.axes.iter() {
                hash_str(&mut hasher, axis);
            }
            for arm in group.arms.iter() {
                match arm {
                    VariantGroupArm::Unit {
                        name,
                    } => {
                        hash_str(&mut hasher, "unit");
                        hash_str(&mut hasher, name);
                    },
                    VariantGroupArm::Product {
                        name,
                        fields,
                    } => {
                        hash_str(&mut hasher, "product");
                        hash_str(&mut hasher, name);
                        for (field, field_type) in fields.iter() {
                            hash_str(&mut hasher, field);
                            hash_str(&mut hasher, field_type);
                        }
                    },
                }
            }
        }

        hash_str(&mut hasher, "helpers");
        for (name, helper) in self.helpers.iter() {
            hash_str(&mut hasher, name);
            hash_str(&mut hasher, &helper.parameter.to_string());
            for (variant, value) in helper.values.iter() {
                hash_str(&mut hasher, variant);
                hasher.update(&value.to_le_bytes());
            }
        }

        hash_str(&mut hasher, "variant group paths");
        for (name, path) in self.variant_group_paths.iter() {
            hash_str(&mut hasher, name);
            hash_str(&mut hasher, path);
        }

        hasher.finalize()
    }

    pub fn full_path_for(
        &self,
        short_name: &str,
    ) -> Option<&str> {
        self.short_name_to_entry.get(short_name).map(|entry| &*entry.path)
    }

    /// Variant names and discriminants of an enum gpu type, in declaration order.
    pub fn variants_for(
        &self,
        short_name: &str,
    ) -> Option<&[(Box<str>, u32)]> {
        self.short_name_to_entry.get(short_name).map(|entry| &*entry.variants)
    }

    pub fn variant_groups(&self) -> &[GpuTypeVariantGroup] {
        &self.variant_groups
    }

    /// Generated accessors that shader CONSTRAINTs may call.
    pub fn helpers(&self) -> &constraint_expr::Helpers {
        &self.helpers
    }

    pub fn variant_group_path(
        &self,
        name: &str,
    ) -> Option<&str> {
        self.variant_group_paths.get(name).map(|path| &**path)
    }

    #[allow(dead_code)]
    pub fn kind_for(
        &self,
        short_name: &str,
    ) -> Option<GpuTypeKind> {
        self.short_name_to_entry.get(short_name).map(|entry| entry.kind)
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
