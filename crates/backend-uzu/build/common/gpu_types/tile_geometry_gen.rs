//! Rust half of the tile geometry emission — see [`super::tile_geometry`] for the Metal
//! half. Both read the same parsed variant names, so they cannot disagree.

use std::{env, path::PathBuf};

use anyhow::Context;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::{
    GpuType, GpuTypes,
    tile_geometry::{ACCESSORS, geometries},
};
use crate::common::codegen::write_tokens;

/// Writes inherent accessors for every tile enum into `$OUT_DIR/tile_geometry.rs`, which
/// the gpu_types module includes.
pub fn tile_geometry_gen(gpu_types: &GpuTypes) -> anyhow::Result<()> {
    let impls = gpu_types
        .files
        .iter()
        .flat_map(|file| file.types.iter())
        .filter_map(|gpu_type| match gpu_type {
            GpuType::Enum(gpu_type_enum) => {
                geometries(gpu_type_enum).map(|geometries| tile_impl(gpu_type_enum.name.as_ref(), &geometries))
            },
            _ => None,
        })
        .collect::<Vec<TokenStream>>();

    let out_path = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("tile_geometry.rs");

    write_tokens(
        quote! {
            #(#impls)*
        },
        &out_path,
    )
    .context("cannot write tile geometry")
}

fn tile_impl(
    type_name: &str,
    geometries: &[(&str, super::tile_geometry::TileGeometry)],
) -> TokenStream {
    let type_ident = format_ident!("{type_name}");

    let accessors = ACCESSORS.iter().map(|(rust_name, _, value_of)| {
        let method = format_ident!("{rust_name}");
        let arms = geometries.iter().map(|(variant, geometry)| {
            let variant = format_ident!("{variant}");
            let value = value_of(geometry);
            quote! { Self::#variant => #value }
        });

        quote! {
            pub const fn #method(self) -> u32 {
                match self {
                    #(#arms,)*
                }
            }
        }
    });

    quote! {
        impl #type_ident {
            #(#accessors)*
        }
    }
}
