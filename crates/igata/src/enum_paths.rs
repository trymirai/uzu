use std::collections::{BTreeMap, HashMap, hash_map::Entry};

use anyhow::bail;
use syn::{Type, TypePath, visit_mut::VisitMut};

use crate::{
    constraint_expr,
    gpu_types::{
        GpuType, GpuTypeName, GpuTypePath, GpuTypeVariantGroup, GpuTypes,
        tile_geometry::{ACCESSORS, geometries},
    },
    mangling::snake_case,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuTypeKind {
    Enum,
    OptionSet,
}

#[derive(Clone, Debug)]
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
                                            parameter: constraint_expr::AxisType::Enum(enum_type.name.clone()),
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
                        let path = format!("crate::backends::common::gpu_types::{}::{}", file.name, group.name);
                        if variant_group_paths.insert(group.name.clone(), path.into_boxed_str()).is_some() {
                            bail!("variant group `{}` is duplicated", group.name);
                        }
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

    /// The gpu-type data the generators read, hashed.
    ///
    /// Wrappers, bindings and the manifest are all derived from this data, but a variant
    /// group emits nothing into the Metal headers whose contents the object cache hashes:
    /// editing one leaves every other cache input identical, so the caches would serve an
    /// `.air` and a `dsl.rs` built for a variant set that no longer exists. Mixing this
    /// in is what makes such an edit visible to them.
    ///
    /// Hashing the `Debug` rendering means the derive decides what is covered, so a field
    /// added to any of these types is covered without anyone remembering to hash it. The
    /// cost of that is a formatting change spuriously rebuilding once. Struct fields and
    /// option-set members are deliberately not here: they reach only the Metal headers,
    /// which the object cache already hashes as `#include` dependencies.
    pub fn semantic_fingerprint(&self) -> blake3::Hash {
        // Ordered, because a HashMap's iteration order is not.
        let entries: BTreeMap<&str, &GpuTypeEntry> =
            self.short_name_to_entry.iter().map(|(name, entry)| (name.as_ref(), entry)).collect();

        blake3::hash(
            format!("{entries:?}{:?}{:?}{:?}", self.variant_groups, self.helpers, self.variant_group_paths).as_bytes(),
        )
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
