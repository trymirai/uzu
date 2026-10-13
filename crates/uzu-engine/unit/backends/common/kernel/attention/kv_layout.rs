use uzu_engine_macros::uzu_test;

const TILE_KEYS: u32 = 32;
const TILE_DIMS: u32 = 16;

/// The lane holding channels `4 * group..4 * group + 4` of fragment row `row` (inverse of the relaxed MXU lane map).
fn mma_lane(
    row: u32,
    group: u32,
) -> u32 {
    16 * (row >> 2) + 8 * (group >> 1) + 4 * ((row >> 1) & 1) + 2 * (row & 1) + (group & 1)
}

pub struct Int8KVLayout {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub padded_capacity: u32,
}

impl Int8KVLayout {
    pub fn scale_count(&self) -> usize {
        (self.kv_heads * self.padded_capacity) as usize
    }

    pub fn code_bytes(&self) -> usize {
        self.scale_count() * self.head_dim as usize
    }

    pub fn k_code_index(
        &self,
        key: u32,
        head: u32,
        channel: u32,
    ) -> usize {
        let row = key % TILE_KEYS;
        let lane = mma_lane(row % 8, channel % TILE_DIMS / 4);
        ((key / TILE_KEYS * self.kv_heads + head) * TILE_KEYS * self.head_dim) as usize
            + (channel / TILE_DIMS * TILE_KEYS * TILE_DIMS + lane * 16 + row / 8 * 4 + channel % 4) as usize
    }

    pub fn v_code_index(
        &self,
        key: u32,
        head: u32,
        channel: u32,
    ) -> usize {
        let (row, slice_channel) = (key % TILE_KEYS, channel % 64);
        let fragment = slice_channel / TILE_DIMS;
        let lane = mma_lane(row % 8, slice_channel % TILE_DIMS / 4);
        let fragment_pair = 4 * (channel / 64) + 2 * (row / 16) + fragment / 2;
        ((key / TILE_KEYS * self.kv_heads + head) * TILE_KEYS * self.head_dim) as usize
            + (fragment_pair * TILE_KEYS * TILE_DIMS + lane * 16 + fragment % 2 * 8 + row % 16 / 8 * 4 + channel % 4)
                as usize
    }
}

#[uzu_test]
fn code_planes_are_bijections() {
    let layout = Int8KVLayout {
        head_dim: 256,
        kv_heads: 2,
        padded_capacity: 96,
    };
    for index in [Int8KVLayout::k_code_index, Int8KVLayout::v_code_index] {
        let mut bytes: Vec<usize> = (0..96 * 2 * 256).map(|i| index(&layout, i / 512, i / 256 % 2, i % 256)).collect();
        bytes.sort_unstable();
        assert!(bytes.into_iter().eq(0..layout.code_bytes()));
    }
}
