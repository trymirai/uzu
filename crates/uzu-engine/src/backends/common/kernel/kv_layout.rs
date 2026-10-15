use half::bf16;

use crate::backends::common::gpu_types::kv_layout;

pub struct Int8KVLayout {
    pub head_dim: u32,
    pub num_kv_heads: u32,
}

impl Int8KVLayout {
    pub fn new(
        head_dim: u32,
        num_kv_heads: u32,
    ) -> Self {
        Self {
            head_dim,
            num_kv_heads,
        }
    }

    /// Scales: [key block][head][key in block].
    #[cfg(backend = "cpu")]
    pub fn scale_index(
        &self,
        key: u32,
        head: u32,
    ) -> usize {
        ((key / kv_layout::TILE_KEYS * self.num_kv_heads + head) * kv_layout::TILE_KEYS + key % kv_layout::TILE_KEYS)
            as usize
    }

    pub fn code_bytes(
        &self,
        keys: u32,
    ) -> usize {
        keys.next_multiple_of(kv_layout::TILE_KEYS) as usize * (self.num_kv_heads * self.head_dim) as usize
    }

    pub fn scale_bytes(
        &self,
        keys: u32,
    ) -> usize {
        self.code_bytes(keys) / self.head_dim as usize * size_of::<bf16>()
    }

    #[cfg(backend = "cpu")]
    pub fn k_code_index(
        &self,
        key: u32,
        head: u32,
        channel: u32,
    ) -> usize {
        let row = key % kv_layout::TILE_KEYS;
        let lane = mma_lane(row % kv_layout::GROUP_ROWS, channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS);
        let lane_offset = row / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS + channel % kv_layout::LANE_COLS;
        ((key / kv_layout::TILE_KEYS * self.num_kv_heads + head) * kv_layout::TILE_KEYS * self.head_dim) as usize
            + (channel / kv_layout::TILE_DIMS * kv_layout::FRAGMENT_BYTES + lane * kv_layout::LANE_BYTES + lane_offset)
                as usize
    }

    #[cfg(backend = "cpu")]
    pub fn v_code_index(
        &self,
        key: u32,
        head: u32,
        channel: u32,
    ) -> usize {
        let (row, slice_channel) = (key % kv_layout::TILE_KEYS, channel % kv_layout::SLICE_DIMS);
        let tile = slice_channel / kv_layout::TILE_DIMS;
        let tiles = kv_layout::SLICE_DIMS / kv_layout::TILE_DIMS;
        let half_rows = kv_layout::TILE_KEYS / 2;
        let lane = mma_lane(row % kv_layout::GROUP_ROWS, slice_channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS);
        let pair = tiles * (channel / kv_layout::SLICE_DIMS) + tiles / 2 * (row / half_rows) + tile / 2;
        let lane_offset = tile % 2 * (kv_layout::LANE_BYTES / 2)
            + row % half_rows / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS
            + channel % kv_layout::LANE_COLS;
        ((key / kv_layout::TILE_KEYS * self.num_kv_heads + head) * kv_layout::TILE_KEYS * self.head_dim) as usize
            + (pair * kv_layout::FRAGMENT_BYTES + lane * kv_layout::LANE_BYTES + lane_offset) as usize
    }
}

#[cfg(backend = "cpu")]
fn mma_lane(
    row: u32,
    group: u32,
) -> u32 {
    // Lane bits: row[2], group[1], row[1], row[0], group[0].
    ((row >> 2) << 4) | ((group >> 1) << 3) | (((row >> 1) & 1) << 2) | ((row & 1) << 1) | (group & 1)
}
