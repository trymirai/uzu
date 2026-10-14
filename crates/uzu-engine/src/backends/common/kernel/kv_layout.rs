pub const TILE_KEYS: u32 = 32;
pub const KV_ROTATION_SIGN_BITS: [u32; 8] =
    [0x9e3779b9, 0x85ebca6b, 0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0xbf58476d, 0x94d049bb, 0xd6e8feb8];
const TILE_DIMS: u32 = 16;
const SLICE_DIMS: u32 = 64;

pub struct Int8KVLayout {
    pub head_dim: u32,
    pub num_kv_heads: u32,
    pub padded_capacity: u32,
}

impl Int8KVLayout {
    pub fn new(
        head_dim: u32,
        num_kv_heads: u32,
        capacity: u32,
    ) -> Self {
        Self {
            head_dim,
            num_kv_heads,
            padded_capacity: capacity.next_multiple_of(TILE_KEYS),
        }
    }

    pub fn scale_count(&self) -> usize {
        (self.num_kv_heads * self.padded_capacity) as usize
    }

    pub fn code_bytes(&self) -> usize {
        self.scale_count() * self.head_dim as usize
    }

    /// Scales are `[key tile][kv head][TILE_KEYS keys]`, so a prefix of keys is a byte prefix.
    pub fn scale_index(
        &self,
        key: u32,
        head: u32,
    ) -> usize {
        ((key / TILE_KEYS * self.num_kv_heads + head) * TILE_KEYS + key % TILE_KEYS) as usize
    }

    pub fn k_code_index(
        &self,
        key: u32,
        head: u32,
        channel: u32,
    ) -> usize {
        let row = key % TILE_KEYS;
        let lane = mma_lane(row % 8, channel % TILE_DIMS / 4);
        ((key / TILE_KEYS * self.num_kv_heads + head) * TILE_KEYS * self.head_dim) as usize
            + (channel / TILE_DIMS * TILE_KEYS * TILE_DIMS + lane * 16 + row / 8 * 4 + channel % 4) as usize
    }

    pub fn v_code_index(
        &self,
        key: u32,
        head: u32,
        channel: u32,
    ) -> usize {
        let (row, slice_channel) = (key % TILE_KEYS, channel % SLICE_DIMS);
        let fragment = slice_channel / TILE_DIMS;
        let lane = mma_lane(row % 8, slice_channel % TILE_DIMS / 4);
        let fragment_pair = 4 * (channel / SLICE_DIMS) + 2 * (row / 16) + fragment / 2;
        ((key / TILE_KEYS * self.num_kv_heads + head) * TILE_KEYS * self.head_dim) as usize
            + (fragment_pair * TILE_KEYS * TILE_DIMS + lane * 16 + fragment % 2 * 8 + row % 16 / 8 * 4 + channel % 4)
                as usize
    }
}

fn mma_lane(
    row: u32,
    group: u32,
) -> u32 {
    16 * (row >> 2) + 8 * (group >> 1) + 4 * ((row >> 1) & 1) + 2 * (row & 1) + (group & 1)
}
