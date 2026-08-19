#[derive(Clone, Copy)]
pub struct Cell {
    pub layer: &'static str,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub bits: u32,
    pub group_size: u32,
}

const LAYERS: &[(&str, u32, u32)] =
    &[("qkv", 2560, 6144), ("o", 4096, 2560), ("gate", 2560, 4096), ("up", 2560, 18432), ("down", 9216, 2560)];

const MS: &[u32] = &[1, 4, 8, 16, 64];

pub const QUANTIZATIONS: &[(u32, u32)] = &[(4, 32), (4, 64), (8, 32), (8, 64)];

pub fn shapes(
    bits: u32,
    group_size: u32,
) -> impl Iterator<Item = Cell> {
    LAYERS.iter().flat_map(move |&(layer, k, n)| {
        MS.iter().map(move |&m| Cell {
            layer,
            m,
            k,
            n,
            bits,
            group_size,
        })
    })
}
