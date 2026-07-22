//! Tile dimensions recovered from enum variant names.
//!
//! A `GemmTiling` variant is named `Tile{M}x{N}x{K}_Simdgroups{SM}x{SN}` and its
//! dimensions were then restated by hand twice — once as Rust match arms, once as Metal
//! constexpr functions — so a tiling's name could disagree with its geometry in two
//! places at once. Parse the name here instead and emit both.

use super::GpuTypeEnum;

#[derive(Debug, Clone, Copy)]
pub struct TileGeometry {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub simdgroups_m: u32,
    pub simdgroups_n: u32,
}

/// The accessors generated for every tile enum: Rust method name, Metal function suffix,
/// and how to read the value.
pub const ACCESSORS: [(&str, &str, fn(&TileGeometry) -> u32); 5] = [
    ("block_m", "block_m", |g| g.block_m),
    ("block_n", "block_n", |g| g.block_n),
    ("block_k", "block_k", |g| g.block_k),
    ("simdgroups_m", "simdgroups_per_row", |g| g.simdgroups_m),
    ("simdgroups_n", "simdgroups_per_column", |g| g.simdgroups_n),
];

/// Predicates a tile's geometry decides, generated for both languages beside the
/// accessors: name, and how to read the answer.
///
/// The tiles that run on the matrix units are exactly the ones staging a 256-deep K
/// block, which is why they need no separate flag to identify them -- but that rule then
/// has to live somewhere, and here it is stated once for the shader and the host both.
pub const PREDICATES: [(&str, fn(&TileGeometry) -> bool); 1] = [("use_mxu", |g| g.block_k == 256)];

impl TileGeometry {
    pub fn parse(variant: &str) -> Option<Self> {
        let (blocks, simdgroups) = variant.strip_prefix("Tile")?.split_once("_Simdgroups")?;

        let [block_m, block_n, block_k] = parse_dimensions::<3>(blocks)?;
        let [simdgroups_m, simdgroups_n] = parse_dimensions::<2>(simdgroups)?;

        Some(Self {
            block_m,
            block_n,
            block_k,
            simdgroups_m,
            simdgroups_n,
        })
    }
}

fn parse_dimensions<const N: usize>(text: &str) -> Option<[u32; N]> {
    let parsed = text.split('x').map(|value| value.parse().ok()).collect::<Option<Vec<u32>>>()?;
    parsed.try_into().ok()
}

/// The geometry of every variant, or `None` if this enum does not name tiles at all.
///
/// All-or-nothing on purpose: a tile enum with one unparseable variant is a typo in that
/// variant, not an enum that happens to be something else.
pub fn geometries(gpu_type_enum: &GpuTypeEnum) -> Option<Vec<(&str, TileGeometry)>> {
    if gpu_type_enum.variants.is_empty() {
        return None;
    }

    gpu_type_enum
        .variants
        .iter()
        .map(|variant| Some((variant.name.as_ref(), TileGeometry::parse(&variant.name)?)))
        .collect()
}

/// `GemmTiling` -> `gemm_tiling`, the prefix its Metal accessors already use.
pub fn metal_prefix(type_name: &str) -> String {
    let mut prefix = String::new();
    for (index, character) in type_name.char_indices() {
        if character.is_uppercase() && index != 0 {
            prefix.push('_');
        }
        prefix.extend(character.to_lowercase());
    }
    prefix
}
